# AI Security on GKE — Worked Examples

Correct-in-shape manifests to imitate for the highest-impact controls. Verify `apiVersion`s, managed-product
feature availability, and Workload Identity binding flow against **current docs** for your GKE version — these
move. This is defensive configuration: isolate and constrain untrusted AI workloads.

---

## 1. gVisor `RuntimeClass` + sandboxed Pod (untrusted code/tool execution)

Use this shape for an agent **code interpreter** or any tool that runs model-influenced code. The Pod runs in
a gVisor sandbox (requires a GKE Sandbox–enabled node pool), holds **no credentials**, has **no egress** (the
NetworkPolicy in §2 enforces that), a read-only root filesystem, all capabilities dropped, and resource caps.

```yaml
# RuntimeClass for gVisor (present on clusters with GKE Sandbox enabled).
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
---
apiVersion: v1
kind: Pod
metadata:
  name: code-interpreter
  namespace: agent-sandbox
  labels:
    app: code-interpreter
spec:
  runtimeClassName: gvisor          # run in the gVisor sandbox
  automountServiceAccountToken: false   # no K8s API token in an untrusted sandbox
  serviceAccountName: sandbox-no-perms  # SA with zero RBAC bindings
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    fsGroup: 10001
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: runner
      image: us-docker.pkg.dev/example/agent/code-runner@sha256:<pin-by-digest>  # digest, not tag
      imagePullPolicy: Always
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      resources:
        requests: { cpu: "250m", memory: "512Mi", ephemeral-storage: "256Mi" }
        limits:   { cpu: "1",    memory: "1Gi",   ephemeral-storage: "1Gi" }   # cap to blunt Model DoS
      volumeMounts:
        - name: scratch
          mountPath: /tmp           # only writable surface
        - name: workspace
          mountPath: /workspace
      env:
        - name: PYTHONUNBUFFERED
          value: "1"
  volumes:
    - name: scratch
      emptyDir: { sizeLimit: 256Mi }
    - name: workspace
      emptyDir: { sizeLimit: 512Mi }
  # Pin to the sandbox node pool; tolerate its taint.
  nodeSelector:
    sandbox.gke.io/runtime: gvisor
  tolerations:
    - key: sandbox.gke.io/runtime
      operator: Equal
      value: gvisor
      effect: NoSchedule
```

Notes:
- **Ephemeral & per-session.** Create a fresh Pod per task/session; never reuse across tenants.
- **No mounted secrets.** If the tool needs an external API, broker it through a separate, audited,
  least-privilege service — don't hand the sandbox a credential.
- Enforce the runtimeClass and these securityContext fields cluster-wide with a Gatekeeper/Kyverno policy so
  the sandbox namespace can't deploy an un-sandboxed Pod.

---

## 2. Default-deny + controlled-egress `NetworkPolicy`

Two policies: a namespace-wide **default-deny** (ingress + egress), then a narrow **allow** for an inference
or agent Pod's required flows only. Egress is the data-exfiltration channel — open it by exception, never by
default. (Requires a NetworkPolicy-capable CNI / Dataplane.)

```yaml
# 2a. Deny ALL ingress and egress in the namespace (the safe default).
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: inference
spec:
  podSelector: {}                 # all pods
  policyTypes: ["Ingress", "Egress"]
  # no ingress/egress rules => deny everything
---
# 2b. Allow ONLY what the inference/agent workload needs.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: inference-allowed-egress
  namespace: inference
spec:
  podSelector:
    matchLabels:
      app: llm-inference
  policyTypes: ["Ingress", "Egress"]
  ingress:
    # Accept traffic only from the gateway namespace (guardrails terminate there).
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: gke-inference-gateway
      ports:
        - port: 8080
          protocol: TCP
  egress:
    # DNS (kube-dns) — required for any name resolution.
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP
    # The vector DB / RAG store, by label, inside the cluster.
    - to:
        - podSelector:
            matchLabels:
              app: vector-db
      ports:
        - port: 6333
          protocol: TCP
    # An explicitly approved external API range (prefer an egress proxy you can log/filter).
    - to:
        - ipBlock:
            cidr: 203.0.113.0/24   # replace with the approved destination(s)
      ports:
        - port: 443
          protocol: TCP
```

Notes:
- A **sandbox** Pod (§1) gets only `default-deny-all` and **no** allow policy — it should reach nothing.
- For internet-bound agent tools, route egress through an **egress proxy/allowlist** so every outbound
  request is logged and policy-checked, rather than opening a raw CIDR.
- `ipBlock` matches post-SNAT addresses; for FQDN-based egress control use an egress proxy or a CNI/feature
  that supports FQDN policy (verify support).

---

## 3. Workload Identity least-privilege (no static keys)

**Principle:** each workload gets its own Kubernetes service account bound to its own least-privilege cloud
IAM principal via **Workload Identity Federation**; Pods receive short-lived, scoped credentials. **Never**
create or mount a static service-account JSON key. The inference server, the RAG indexer, and the agent's
tools are **different** identities — avoid one all-powerful "agent" SA (that is excessive agency).

```yaml
# Distinct KSAs, distinct cloud identities, distinct scopes.
apiVersion: v1
kind: ServiceAccount
metadata:
  name: llm-inference          # may read model weights bucket (read-only) only
  namespace: inference
  # Binding annotation/flow is version-specific — verify the current Workload Identity setup.
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rag-indexer            # may write the index; cannot read secrets or call agent tools
  namespace: inference
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sandbox-no-perms       # zero permissions; used by §1 sandbox with token automount OFF
  namespace: agent-sandbox
```

Least-privilege checklist for this workload:

- **No static keys.** Bind the KSA to a cloud identity via Workload Identity Federation; verify the exact
  annotation/binding command for your GKE version — **verify against current docs**.
- **Scope tightly.** The inference SA gets read-only on the weights bucket and nothing else. The indexer gets
  write on the index and nothing else. Neither can read Secret Manager entries it doesn't use.
- **Map IAM ↔ RBAC.** Grant only the K8s RBAC each SA needs; most data-plane workloads need **none** and
  should set `automountServiceAccountToken: false`.
- **Secrets** come from Secret Manager at runtime via the scoped identity — never baked into the image, never
  in env in plaintext where avoidable, never logged, never placed in prompts.
- **Audit.** Enable audit logging; alert on any use of a long-lived key (there should be none).
