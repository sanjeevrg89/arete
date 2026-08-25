# GKE Worked Examples

Annotated, correct-in-shape artifacts to imitate. **Verify machine types, accelerator SKUs, topologies,
regions, and flags against current GKE docs and your project's quota before running** — names and
defaults change. Placeholders use `<ANGLE_BRACKETS>`.

---

## 1. Create accelerator node pools (`gcloud container node-pools create`)

### 1a. Multi-host TPU v5p slice (Standard)

```bash
# A multi-host TPU v5p slice node pool. The node pool IS the slice: machine type + topology
# determine chips and host count. 2x2x4 (= 16 chips) over multiple hosts connected by ICI.
# All hosts schedule all-or-nothing; pair with JobSet/LWS + Kueue for the workload.  [[kueue-advanced]]
gcloud container node-pools create tpu-v5p-16 \
  --cluster=<CLUSTER> \
  --location=<REGION> \                      # regional cluster
  --node-locations=<ZONE> \                  # pin to the single zone that has the SKU + quota
  --machine-type=ct5p-hightpu-4t \           # v5p host type (4 TPU chips/host) — VERIFY current name
  --tpu-topology=2x2x4 \                      # slice topology -> total chips; must match the machine type
  --num-nodes=4 \                             # hosts in the slice (chips/topology ÷ chips/host)
  --placement-type=COMPACT \                  # contiguous placement for tight ICI
  --node-labels=team=research \
  --enable-gvnic                              # gVNIC for high-throughput networking

# Pods then select the slice via labels and request chips/host:
#   nodeSelector:
#     cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
#     cloud.google.com/gke-tpu-topology: 2x2x4
#   resources: { limits: { google.com/tpu: "4" } }   # chips per host
```

### 1b. H100 GPU node pool with GPUDirect-TCPX (Standard)

```bash
# A3 (8x H100). For collective-heavy training you also wire multi-networking + the NCCL/TCPX
# installer DaemonSet (separate step) — the pool just provides the hardware + extra NICs + driver.
gcloud container node-pools create a3-h100 \
  --cluster=<CLUSTER> \
  --location=<REGION> \
  --node-locations=<ZONE> \                  # accelerator pools are typically single-zone
  --machine-type=a3-highgpu-8g \             # 8x H100 80GB — VERIFY current family/name
  --accelerator=type=nvidia-h100-80gb,count=8,gpu-driver-version=latest \  # GKE-managed driver
  --ephemeral-storage-local-ssd=count=16 \   # local SSD for scratch/checkpoints + FUSE cache
  --num-nodes=2 \
  --enable-autoscaling --min-nodes=0 --max-nodes=8 \
  --node-taints=nvidia.com/gpu=present:NoSchedule \  # only GPU-tolerating Pods land here
  --node-labels=accelerator=h100 \
  --image-type=COS_CONTAINERD                # Container-Optimized OS

# Spot variant for fault-tolerant training: add `--spot` and a Spot taint; checkpoint frequently.
# Prefer Node Auto-Provisioning + a Custom Compute Class for graceful fallback instead of pinning.
```

> On **Autopilot** you don't create node pools — you request the accelerator on the Pod
> (`cloud.google.com/gke-accelerator: nvidia-h100-80gb`, `nvidia.com/gpu`, optional
> `cloud.google.com/gke-spot`) and GKE provisions the node. A **Custom Compute Class** expresses
> priority/fallback across GPU types in both modes.

---

## 2. Workload Identity Federation for GKE (the right way to call GCP APIs)

No SA keys. Bind a Kubernetes ServiceAccount (KSA) to an IAM identity, then grant that identity the GCP
roles it needs.

```bash
# 0) Cluster must have Workload Identity enabled (default on Autopilot):
#    gcloud container clusters update <CLUSTER> --location <REGION> \
#      --workload-pool=<PROJECT_ID>.svc.id.goog

PROJECT_ID=<PROJECT_ID>
NAMESPACE=research
KSA=trainer                 # Kubernetes ServiceAccount the Pods run as
GSA=trainer-gsa             # (optional) IAM service account to impersonate

# 1) Create the KSA
kubectl create serviceaccount $KSA --namespace $NAMESPACE

# 2a) DIRECT binding (no GSA): grant the KSA principal a role directly. Preferred when possible.
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --role=roles/storage.objectViewer \
  --member="principal://iam.googleapis.com/projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA}"

# 2b) OR impersonate an IAM SA (when you need to reuse an existing GSA's grants):
gcloud iam service-accounts create $GSA
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --role=roles/storage.objectViewer \
  --member="serviceAccount:${GSA}@${PROJECT_ID}.iam.gserviceaccount.com"
# Let the KSA impersonate the GSA:
gcloud iam service-accounts add-iam-policy-binding ${GSA}@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA}]"
# And annotate the KSA to point at the GSA:
kubectl annotate serviceaccount $KSA --namespace $NAMESPACE \
  iam.gke.io/gcp-service-account=${GSA}@${PROJECT_ID}.iam.gserviceaccount.com
```

```yaml
# 3) Run the Pod as that KSA — it gets GCP credentials via the metadata server, no keys mounted.
apiVersion: v1
kind: Pod
metadata:
  name: trainer
  namespace: research
spec:
  serviceAccountName: trainer        # <- the KSA bound above
  containers:
    - name: trainer
      image: <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/trainer:latest
      # SDKs/gcloud/gsutil inside here authenticate as the bound identity automatically.
```

---

## 3. GCS FUSE CSI volume on a Pod (datasets / checkpoints)

Mount a GCS bucket as a volume. Requires the **GCS FUSE CSI driver addon**
(`--addons GcsFuseCsiDriver`) and bucket access via **Workload Identity** (section 2) — grant the KSA
`roles/storage.objectAdmin` (or narrower) on the bucket.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fuse-trainer
  namespace: research
  annotations:
    gke-gcsfuse/volumes: "true"                 # opt the Pod into the CSI sidecar injection
    # Optional sidecar sizing for heavy I/O:
    gke-gcsfuse/cpu-limit: "0"                   # "0" = unlimited; tune for throughput
    gke-gcsfuse/memory-limit: "0"
    gke-gcsfuse/ephemeral-storage-limit: "0"     # cache can be large; back ephemeral with local SSD
spec:
  serviceAccountName: trainer                    # Workload Identity → bucket access, no keys
  containers:
    - name: trainer
      image: <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/trainer:latest
      volumeMounts:
        - name: data
          mountPath: /data
          readOnly: true                         # datasets read-only
        - name: ckpt
          mountPath: /checkpoints                 # checkpoints read-write
  volumes:
    - name: data
      csi:
        driver: gcsfuse.csi.storage.gke.io
        readOnly: true
        volumeAttributes:
          bucketName: <DATASET_BUCKET>
          mountOptions: "implicit-dirs"           # plus caching opts for throughput, e.g.:
          # file caching / metadata caching dramatically speed small-file dataset reads —
          # set fileCacheCapacity / metadataCacheTtlSecs (and a local-SSD-backed cache dir) as needed.
          fileCacheCapacity: "100Gi"
          metadataStatCacheCapacity: "-1"         # -1 = unbounded stat cache (tune to dataset)
    - name: ckpt
      csi:
        driver: gcsfuse.csi.storage.gke.io
        volumeAttributes:
          bucketName: <CHECKPOINT_BUCKET>
          mountOptions: "implicit-dirs"
```

> For "100 nodes load the same 200GB checkpoint fast," prefer **Hyperdisk ML** (read-only multi-attach)
> over cold GCS FUSE reads. For high-throughput parallel scratch, use **Parallelstore CSI**. Stage
> in-step scratch and checkpoint writes on **local SSD**, then async-flush to GCS.

---

## 4. PodMonitoring for Managed Service for Prometheus (per-team metrics)

Scrape an ML serving/training Pod's `/metrics` with Managed Prometheus. Metrics land on the
`k8s_container` resource carrying `namespace_name` → per-team attribution.

```yaml
apiVersion: monitoring.googleapis.com/v1
kind: PodMonitoring
metadata:
  name: vllm-metrics
  namespace: serving           # scoped to this namespace -> namespace_name attribution
spec:
  selector:
    matchLabels:
      app: vllm
  endpoints:
    - port: metrics            # named container port exposing Prometheus metrics
      interval: 15s
      path: /metrics
# Use ClusterPodMonitoring (cluster-scoped) for fleet/infra exporters like DCGM.
# Enable DCGM (GPU) / TPU metrics on the cluster for accelerator signals:
#   GPU -> duty_cycle ;  TPU -> tensorcore_utilization  (different names, same intent).
```
