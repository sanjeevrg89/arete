# Ray on Kubernetes — Worked Examples

Annotated, runnable-in-spirit manifests and snippets to imitate. KubeRay CRD fields move between
releases — **verify `apiVersion`, field names, and GCS-FT/autoscaler keys against the docs for your exact
KubeRay/Ray version.** Replace image tags, namespaces, node selectors, and resource sizes for your env.

---

## 1. RayCluster (long-lived, GPU workers, autoscaling, GCS FT)

A shared cluster: CPU head (no compute), an autoscaling GPU worker group, GCS fault tolerance via
external Redis.

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-shared
  namespace: ray
  annotations:
    ray.io/ft-enabled: "true"            # enable GCS fault tolerance (verify key for your version)
spec:
  rayVersion: "2.x"                       # MUST match the ray image tag below
  enableInTreeAutoscaler: true
  autoscalerOptions:
    idleTimeoutSeconds: 300               # scale a worker down after 5m idle
    upscalingMode: Default

  headGroupSpec:
    rayStartParams:
      num-cpus: "0"                       # keep tasks/actors OFF the head
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray:2.x      # pin; must match rayVersion
            resources:                     # size for GCS + dashboard + object-owner metadata
              requests: { cpu: "4", memory: "16Gi" }
              limits:   { cpu: "4", memory: "16Gi" }
            env:
              # External Redis for GCS FT (use a Secret; verify exact env keys for your version)
              - name: RAY_REDIS_ADDRESS
                value: "redis-master.ray.svc.cluster.local:6379"
              - name: REDIS_PASSWORD
                valueFrom: { secretKeyRef: { name: ray-redis, key: password } }
            ports:
              - { containerPort: 6379, name: gcs }
              - { containerPort: 10001, name: client }
              - { containerPort: 8265, name: dashboard }
            volumeMounts:
              - { name: ray-logs, mountPath: /tmp/ray }
        volumes:
          - { name: ray-logs, emptyDir: {} }
        nodeSelector:
          cloud.google.com/compute-class: "cpu-pool"   # head on a CPU pool

  workerGroupSpecs:
    - groupName: gpu-a100
      replicas: 1
      minReplicas: 1                       # >0 if you want warm GPUs; 0 to scale to zero
      maxReplicas: 16
      rayStartParams:
        num-gpus: "1"                      # logical resource MUST match the pod request below
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray:2.x-gpu
              resources:
                requests: { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }
                limits:   { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }  # requests==limits for GPU
              volumeMounts:
                - { name: spill, mountPath: /tmp/ray }   # spill to FAST LOCAL SSD
          volumes:
            - name: spill
              ephemeral:                   # local SSD-backed; do NOT spill to slow network disk
                volumeClaimTemplate:
                  spec: { accessModes: ["ReadWriteOnce"], resources: { requests: { storage: 200Gi } } }
          nodeSelector:
            cloud.google.com/gke-accelerator: "nvidia-tesla-a100"
          tolerations:
            - { key: "nvidia.com/gpu", operator: "Exists", effect: "NoSchedule" }
```

---

## 2. RayJob (run-to-completion, ephemeral cluster, Kueue-ready)

Creates a cluster, runs the entrypoint, tears down. `suspend` + a queue label let **Kueue** gate
admission for quota/gang scheduling (`[[kueue-advanced]]`).

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: train-llm
  namespace: ray
  labels:
    kueue.x-k8s.io/queue-name: gpu-queue   # Kueue local queue (verify label for your Kueue version)
spec:
  entrypoint: "python /home/ray/train.py --epochs 3"
  shutdownAfterJobFinishes: true           # delete the cluster when the job ends
  ttlSecondsAfterFinished: 600
  suspend: true                            # Kueue flips this to false on admission
  # submissionMode: K8sJobMode             # verify available modes for your version
  rayClusterSpec:
    rayVersion: "2.x"
    headGroupSpec:
      rayStartParams: { num-cpus: "0" }
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.x
              resources:
                requests: { cpu: "4", memory: "16Gi" }
                limits:   { cpu: "4", memory: "16Gi" }
    workerGroupSpecs:
      - groupName: gpu
        replicas: 4                         # gang: Ray Train will form a PG across these 4 workers
        minReplicas: 4
        maxReplicas: 4
        rayStartParams: { num-gpus: "1" }
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray:2.x-gpu
                resources:
                  requests: { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }
                  limits:   { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }
            nodeSelector: { cloud.google.com/gke-accelerator: "nvidia-tesla-a100" }
            tolerations:
              - { key: "nvidia.com/gpu", operator: "Exists", effect: "NoSchedule" }
```

---

## 3. RayService (HA serving, zero-downtime upgrades, autoscaling)

Ray Serve behind a RayService. Spec changes trigger a blue/green upgrade (new cluster comes up healthy,
then traffic cuts over). Enable GCS FT in `rayClusterConfig` for production.

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: text-model
  namespace: ray
spec:
  serveConfigV2: |                          # verify schema for your Serve version
    applications:
      - name: text
        import_path: app:deployment_graph   # module:variable for the Serve app
        route_prefix: /
        deployments:
          - name: Model
            num_replicas: auto              # Serve autoscaling (param names vary by version — verify)
            autoscaling_config:
              min_replicas: 2               # >0: keep warm replicas, no cold start on spikes
              max_replicas: 20
              target_ongoing_requests: 8
            ray_actor_options:
              num_gpus: 1                    # one GPU per replica; matches pod GPU request
  rayClusterConfig:
    rayVersion: "2.x"
    headGroupSpec:
      rayStartParams: { num-cpus: "0", dashboard-host: "0.0.0.0" }
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.x
              ports:
                - { containerPort: 8000, name: serve }    # Serve HTTP proxy
              resources:
                requests: { cpu: "4", memory: "16Gi" }
                limits:   { cpu: "4", memory: "16Gi" }
    workerGroupSpecs:
      - groupName: gpu
        replicas: 2
        minReplicas: 2
        maxReplicas: 20
        rayStartParams: { num-gpus: "1" }
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray:2.x-gpu
                resources:
                  requests: { cpu: "8", memory: "48Gi", nvidia.com/gpu: "1" }
                  limits:   { cpu: "8", memory: "48Gi", nvidia.com/gpu: "1" }
            nodeSelector: { cloud.google.com/gke-accelerator: "nvidia-l4" }
            tolerations:
              - { key: "nvidia.com/gpu", operator: "Exists", effect: "NoSchedule" }
```

---

## 4. Placement group — gang scheduling for multi-host work

Reserve resources atomically so all ranks start together. `STRICT_SPREAD` = one bundle per node (typical
multi-host training); `PACK`/`STRICT_PACK` = colocate.

```python
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

ray.init()  # connects to the cluster (RAY_ADDRESS / ray client)

NUM_WORKERS = 4
# One bundle per worker; STRICT_SPREAD forces each onto a distinct node (one GPU host each).
pg = placement_group(
    bundles=[{"GPU": 1, "CPU": 8} for _ in range(NUM_WORKERS)],
    strategy="STRICT_SPREAD",
)
ray.get(pg.ready())  # BLOCKS until the whole gang is reserved — no partial allocation / deadlock

@ray.remote(num_gpus=1, num_cpus=8)
class Worker:
    def rank_info(self):
        import os
        return os.environ.get("CUDA_VISIBLE_DEVICES")

# Pin each actor into its own bundle of the placement group.
workers = [
    Worker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i
        )
    ).remote()
    for i in range(NUM_WORKERS)
]

print(ray.get([w.rank_info.remote() for w in workers]))
remove_placement_group(pg)  # always release the reservation when done
```

> In practice, **Ray Train / Tune / Serve create and manage placement groups for you** — only hand-write
> a PG for raw task/actor gang workloads. Always `pg.ready()` (or let the library gang-admit) before
> launching work, and always remove the PG to free the reservation.
