# AI Security on GKE — Defense-in-Depth for LLM Workloads

This is the full reference for securing AI/LLM workloads (inference, RAG, agents) on Kubernetes/GKE. It is a
**defensive** guide: the goal is to protect AI platforms, not attack them. The organizing principle is
**defense in depth** — assume any single control fails, and put an independent control at every layer.

Fast-moving and proprietary items (managed-product capabilities, exact API surfaces, model names) are flagged
**verify against current docs**. Do not rely on a specific feature flag without confirming it exists today.

---

## 1. Mental model

A traditional web service trusts its own code and distrusts the client. An LLM platform must distrust **four**
boundaries simultaneously:

1. **The prompt** (the user can try to jailbreak or inject).
2. **The model output** (it is a probabilistic text generator — never treat its output as a safe command,
   safe SQL, safe HTML, or trustworthy fact).
3. **Retrieved/tool/web content** (RAG documents, search results, tool responses, emails, and pages the agent
   reads can carry attacker-planted instructions — *indirect prompt injection*).
4. **The supply chain** (model weights and container images can execute code or carry backdoors).

The model is not a security boundary. It is a **confused-deputy machine**: it will faithfully follow whatever
text ends up in its context, regardless of where that text came from. Security therefore lives **around** the
model — at the gateway, in the sandbox, in IAM/RBAC, in the network policy — not inside the prompt. "Just tell
the model not to do X" is never a control; it is a hint an attacker will override.

Defense in depth means: input filter **and** output filter **and** least-privilege tools **and** sandbox
**and** egress control **and** runtime detection. Any one will be bypassed eventually; the stack is what holds.

---

## 2. Threat model — OWASP LLM Top 10 → control

The OWASP Top 10 for LLM Applications is the canonical taxonomy (verify the current year's revision — the list
is periodically renumbered/renamed). Each risk maps to one or more layers below. Use this as the spine of any
threat-modeling session.

| OWASP LLM risk | What it is | Primary control(s) |
|---|---|---|
| **Prompt injection (direct)** | User crafts input to override system instructions / jailbreak | Input guardrail (classifier), instruction/data separation, least-privilege tools, output filter |
| **Prompt injection (indirect)** | Malicious instructions arrive via RAG docs / web / tool output | Sanitize & delimit untrusted content, output filter, sandbox tools, human-in-loop for high-impact actions |
| **Insecure output handling** | Treating model output as trusted (exec, SQL, HTML, shell, file paths) | Output validation/encoding, no direct exec, parameterized queries, sandbox, allowlist actions |
| **Training-data poisoning** | Tainted fine-tune / RAG corpus alters behavior or implants backdoors | Dataset provenance & integrity, curation, signed datasets, eval gates before promotion |
| **Model DoS** | Token-flood, huge context, recursive/looping agents exhaust resources | Rate limits, token/context caps, request timeouts, resource quotas, loop/step caps |
| **Supply-chain** | Malicious model weights, base images, or libraries | Image signing/verify (Binary Authorization, Sigstore), safetensors over pickle, scanning, SLSA |
| **Sensitive-info disclosure** | Model leaks PII, secrets, other tenants' data, or system prompt | Output PII/secret redaction, input redaction, tenant isolation, minimize secrets in context |
| **Excessive agency** | Agent has more tools/permissions/autonomy than needed | Least-privilege tool scoping, per-action authz, scoped credentials, human approval gates |
| **Overreliance** | Humans/systems trust hallucinated output without verification | Grounding/citations, confidence signals, validation, keep humans in the loop |
| **Model theft** | Exfiltration of weights / extraction via queries | Access control on weights, egress control, Confidential GKE, rate limiting, watermarking (where applicable) |

The two that dominate real incidents on agentic platforms are **indirect prompt injection** and **excessive
agency** — they compound: an injected instruction is only dangerous in proportion to the agency you granted.

---

## 3. Content & model defense (input and output guardrails)

Guardrails are classifiers/filters that sit **before** the prompt reaches the model and **after** the model
generates, ideally enforced at the inference gateway so every path is covered (see `[[gke-inference-gateway]]`).

**Where to place them:**

- **Pre-prompt (ingress):** prompt-injection/jailbreak detection, topic/policy enforcement, input PII
  redaction, request shaping (token caps). Reject or sanitize before spending model compute.
- **Post-generation (egress):** output safety/toxicity classification, PII/secret leak detection, grounding
  checks, and — critically for agents — **validating any proposed tool call / action** before it executes.
- **Per-tool / per-retrieval:** when RAG or a tool returns content, treat that content as untrusted input and
  re-screen it before it re-enters the model's context.

**Options (verify current capabilities and licensing):**

| Option | Type | Notes |
|---|---|---|
| **Model Armor** | Managed (GCP) | Managed prompt-injection/jailbreak detection, sensitive-data & malicious-URL screening, safety filtering; integrate at gateway/app boundary. Verify supported regions/features. |
| **Llama Guard** | OSS model | Input/output safety classifier (Meta). Run as a small sidecar/model behind the gateway. |
| **NeMo Guardrails** | OSS framework (NVIDIA) | Programmable rails (input, output, dialog, retrieval) via Colang; good for structured policy. |
| **Guardrails-AI** | OSS framework | Validators for structure, PII, toxicity, format; pydantic-style output contracts. |
| **Prompt-shield-style classifiers** | Managed/OSS | Dedicated injection/jailbreak detectors; capabilities differ — verify. |

**Design rules:**

- **Separate instructions from data.** Put untrusted content (user/RAG/tool) in a clearly delimited region and
  instruct the model to treat it as data, not commands. Delimiting is a mitigation, not a guarantee — pair it
  with a real classifier and with least-privilege downstream.
- **Fail closed for high-impact paths.** If the guardrail service is down, block irreversible actions; you may
  allow read-only chat to degrade gracefully, but never let a fail-open path execute tools.
- **Filter both directions.** Input-only filtering misses indirect injection and data exfiltration in outputs.
- **Don't put secrets in the system prompt.** It can leak. Inject only what the request needs.
- **Log and rate-limit.** Keep an auditable trail of blocked attempts; many injection campaigns are noisy.

---

## 4. Agent & code sandboxing (untrusted execution)

Agentic apps run model-chosen code and tools. **Any code path the model can influence must run sandboxed.** The
classic disaster is a "code interpreter" tool that runs LLM-generated Python with cluster network access and a
mounted service-account token.

**Isolation tiers (strongest last):**

| Mechanism | Isolation | Use for |
|---|---|---|
| `seccompProfile: RuntimeDefault` + dropped caps | Syscall/capability narrowing | Baseline on *every* Pod |
| AppArmor / SELinux profile | MAC confinement | Constrain file/exec surface |
| **gVisor (GKE Sandbox)** | User-space kernel; intercepts syscalls | Untrusted code/tool execution — the default for code interpreters and risky tools |
| **Kata Containers / microVMs** | Hardware-virtualized VM per pod | Strong multi-tenant / hostile-workload isolation where gVisor's syscall coverage or compatibility doesn't fit; verify GKE support model |

On GKE, gVisor is exposed via **GKE Sandbox** (and the emerging **Agent Sandbox** concept for agent tool
execution — verify current availability/name). You select it with a `RuntimeClass` (`gvisor`) and a node pool
that has sandboxing enabled. See `examples.md` for a full manifest.

**Hardening checklist for an untrusted-execution Pod:**

- `runtimeClassName: gvisor` (sandboxed node pool).
- `securityContext`: `runAsNonRoot: true`, `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`,
  `capabilities.drop: ["ALL"]`, `seccompProfile.type: RuntimeDefault`.
- **No network by default** — attach a `NetworkPolicy` that denies all egress (and ingress), allow nothing
  unless a specific tool truly needs it. Network is the exfiltration channel.
- **No mounted credentials** — `automountServiceAccountToken: false`; the sandbox should not hold cluster or
  cloud creds. If the tool needs an API, broker it through a separate, audited, least-privilege service.
- **Resource caps** — set CPU/memory `limits`, ephemeral-storage limits, and a step/time budget so a runaway
  or recursive agent can't DoS the node (Model DoS).
- **Ephemeral & disposable** — fresh sandbox per session/task; never reuse across tenants; no persistent
  writable state shared between executions.
- For agent frameworks and tool design, see `[[llm-app-agent-frameworks]]`.

---

## 5. Runtime security

Detection complements prevention: assume something slips through and watch the running workload.

- **GKE Security Posture / Container Threat Detection** — managed runtime threat detection (suspicious binary
  execution, reverse shells, crypto-mining patterns) plus workload misconfiguration and vulnerability findings.
  Enable the dashboard and route findings to your SIEM. Verify current detector coverage.
- **Falco** — OSS runtime security (eBPF/kernel syscall monitoring) with a rules engine; deploy as a
  DaemonSet for portable detection and custom rules (e.g., alert on a sandbox Pod attempting outbound
  connections or spawning a shell).
- **Admission control** — stop bad workloads at the door:
  - **Pod Security Admission (PSA)** — enforce the `restricted` profile by default at the namespace level;
    only relax (e.g., for a gVisor node pool that still meets `restricted`) with explicit justification.
  - **Gatekeeper (OPA)** or **Kyverno** — policy-as-code for what PSA can't express: require signed images,
    forbid `:latest`, require `runtimeClassName: gvisor` in the sandbox namespace, require resource limits,
    forbid `hostPath`/`hostNetwork`/privileged, require a default-deny NetworkPolicy per namespace.
- **Image signing & verification** — **Binary Authorization** gates deployment on attestations (built by your
  trusted pipeline, scanned, signed). Pair with **Sigstore (cosign)** for keyless signing and verification. No
  unsigned or unscanned image reaches production.

For cluster-level posture (private clusters, shielded/secure boot nodes, node auto-upgrade), see
`[[gke-master]]` and `[[kubernetes-expert]]`.

---

## 6. Supply chain — models, images, and data

Model artifacts are **executable**, not inert data. The single most important model-supply-chain rule:

> **Never load untrusted model weights from a pickle-based format.** Python pickle / `torch.load` /
> `joblib` / arbitrary `.bin` checkpoints execute arbitrary code during deserialization. A poisoned checkpoint
> downloaded from a public hub can pop a shell the moment you load it. **Prefer `safetensors`** (no code
> execution on load). If you must load pickle, do it only from a source you fully trust, after scanning, in a
> sandbox with no credentials and no egress.

Broader supply-chain controls:

- **Provenance** — track where each model came from (origin, hash, license). Mirror trusted weights into your
  own registry; pin by digest, not by mutable tag. Verify checksums/signatures on download.
- **Scanning** — scan model files (pickle-opcode scanners, malware scanners) and container images (vuln
  scanning) before promotion. Scan dependencies; pin and lock them.
- **SLSA** — adopt SLSA provenance for build artifacts so you can prove an image/model was built by your
  pipeline from known source. Gate deploys on it via Binary Authorization.
- **Dataset integrity** — for fine-tuning/RAG corpora, control and version the source; hash/sign datasets;
  curate against poisoning; run eval gates (including safety/red-team evals) before a model or index is
  promoted. Treat the RAG index as a trust boundary — anyone who can write to it can inject instructions into
  every future answer.

---

## 7. Identity, secrets, network, and data protection

This is where most "AI" incidents actually land — ordinary cloud-security failures around an AI workload.

### Identity & access

- **Workload Identity Federation** — Pods assume a scoped identity and receive **short-lived** credentials; do
  **not** create or mount static service-account JSON keys. Bind each workload's Kubernetes service account to
  a least-privilege IAM principal. (Verify the current Workload Identity setup/binding flow for your GKE
  version.) See `examples.md` for a least-privilege note.
- **Least privilege, IAM ↔ RBAC** — one service account per workload, scoped to exactly the APIs/buckets it
  needs. The inference server, the RAG indexer, and the agent's tools should be **different** identities. Avoid
  one over-permissioned "agent" identity that can do everything the model asks.
- **`automountServiceAccountToken: false`** on anything that doesn't call the Kubernetes API — especially
  sandboxes. A leaked token in the model's context is a lateral-movement primitive.

### Secrets

- Store secrets in **Secret Manager** (or an external KMS-backed store), fetched at runtime via the workload's
  scoped identity — not in env vars baked into images, not in plaintext K8s Secrets where avoidable. Rotate.
- Never place secrets/credentials in prompts, system prompts, tool descriptions, or logs. The model can echo
  them (sensitive-info disclosure).

### Network

- **Default-deny NetworkPolicy** per namespace, then allow only required flows. For an inference/agent Pod,
  egress is the exfiltration channel — a successful prompt injection wants to send data out. Allow only DNS +
  the specific destinations the workload needs (model registry, vector DB, approved APIs). Deny everything
  else. See `examples.md` for the full policy.
- **Private clusters / private nodes** — no public node IPs; control-plane access restricted; egress through a
  controlled NAT/proxy you can log and filter. See `[[gke-master]]`.
- Consider an **egress proxy/allowlist** for agent tools that must reach the internet, so every outbound
  request is logged and policy-checked.

### Data & encryption

- Encryption at rest and in transit by default; manage keys via KMS (CMEK where required).
- **Confidential GKE** — confidential computing (memory encryption / hardware-based isolation) for the most
  sensitive weights or data, reducing exposure to the host/operator. Verify current machine-type and feature
  support. Useful when weights are a regulated or high-value asset (model-theft mitigation).
- **Tenant isolation** in multi-tenant inference — separate namespaces, network policies, identities, and
  ideally separate sandboxes/nodes; never let one tenant's RAG content or session bleed into another's context.

---

## 8. Defense-in-depth mapping (threat → layer → control)

| Layer | Controls | Mitigates |
|---|---|---|
| **Gateway / app** | Input & output guardrails (Model Armor / Llama Guard / NeMo / Guardrails-AI), rate limits, token caps, instruction/data separation | Direct & indirect injection, sensitive-info disclosure, model DoS, insecure output |
| **Agent design** | Least-privilege tools, per-action authz, human-in-loop, step/loop caps, grounding/citations | Excessive agency, overreliance, indirect injection blast radius |
| **Sandbox** | gVisor / Kata, seccomp, AppArmor, drop caps, read-only rootfs, no creds, no egress, resource caps | Insecure output → RCE, code-interpreter abuse, model DoS |
| **Admission** | PSA `restricted`, Gatekeeper/Kyverno, Binary Authorization, Sigstore | Supply-chain, misconfiguration, unsigned images |
| **Runtime** | GKE Security Posture / Container Threat Detection, Falco, audit logging | Post-exploitation detection, all categories |
| **Supply chain** | safetensors over pickle, artifact/image scanning, SLSA, dataset integrity, pin by digest | Supply-chain, training-data poisoning |
| **Identity & secrets** | Workload Identity Federation, least-priv IAM↔RBAC, Secret Manager, no static keys, no auto-mounted tokens | Excessive agency, sensitive-info disclosure, lateral movement |
| **Network** | Default-deny + controlled egress, private clusters, egress proxy/allowlist | Data exfiltration, model theft, C2 |
| **Data** | Encryption + KMS/CMEK, Confidential GKE, tenant isolation | Model theft, sensitive-info disclosure |

---

## 9. AI-platform security checklist

**Content / model defense**
- [ ] Input guardrail (injection/jailbreak/PII) at the gateway, on every path.
- [ ] Output guardrail (toxicity/PII/secret leak) and tool-call validation post-generation.
- [ ] Untrusted RAG/tool/web content delimited and re-screened before re-entering context.
- [ ] No secrets in system prompts; fail-closed for high-impact actions.

**Agent / execution**
- [ ] All untrusted code/tool execution in gVisor (or Kata) sandbox.
- [ ] Sandbox: non-root, read-only rootfs, drop ALL caps, `RuntimeDefault` seccomp, no auto-mounted token,
      resource & step/time caps, ephemeral per session.
- [ ] Least-privilege tools; per-action authz; human approval for irreversible actions; step/loop caps.

**Supply chain**
- [ ] Weights in `safetensors`; pickle only from trusted source, scanned, sandboxed; pinned by digest.
- [ ] Images signed (Binary Authorization / Sigstore), scanned, SLSA provenance; deploy gated on attestation.
- [ ] Datasets/RAG corpora versioned, integrity-checked, eval-gated before promotion.

**Identity / secrets / network / data**
- [ ] Workload Identity Federation; no static SA keys; per-workload least-privilege identities.
- [ ] Secrets in Secret Manager; never logged or placed in prompts.
- [ ] Default-deny NetworkPolicy + controlled egress; private cluster; egress proxy for internet-bound tools.
- [ ] Encryption + KMS/CMEK; Confidential GKE for sensitive weights/data; tenant isolation.

**Runtime / admission**
- [ ] PSA `restricted` by default; Gatekeeper/Kyverno policies enforced.
- [ ] GKE Security Posture / Container Threat Detection and/or Falco enabled; findings to SIEM; audit logging on.

---

## 10. Anti-patterns

- **Trusting model or tool output.** Passing model output straight to `exec`/`eval`/shell, into SQL string
  concatenation, into HTML without encoding, or executing a tool call without validation. Output is untrusted
  input — validate, encode, parameterize, allowlist.
- **No egress controls.** A Pod with open egress turns any successful injection into a data-exfiltration
  channel. Default-deny and allowlist.
- **Running untrusted code unsandboxed.** A code interpreter or agent tool sharing the node's kernel, network,
  and credentials. Always gVisor/Kata, no creds, no egress, caps set.
- **Static service-account keys.** JSON keys in images/Secrets/env. Use Workload Identity Federation;
  short-lived, scoped credentials only.
- **Loading pickle weights from untrusted sources.** Arbitrary code execution on load. Prefer safetensors;
  scan; sandbox.
- **No output filtering.** Input-only guardrails miss indirect injection and PII/secret leakage on the way out.
- **One over-permissioned agent identity.** A single service account that can touch every API the model might
  request — maximal excessive agency. Split identities; scope to least privilege.
- **"Tell the model not to" as a control.** Prompt instructions are not a security boundary; an attacker's
  text overrides yours. Controls live around the model.
- **Mutable image/model tags in production.** Pin by digest; gate on signatures.

---

## 11. Version awareness

The AI-security ecosystem moves fast (it is 2026). Managed-product features (Model Armor capabilities, GKE
Security Posture detectors, Agent Sandbox availability, Confidential GKE machine support), the OWASP LLM Top 10
numbering, and guardrail-model versions all change. **Verify every proprietary or fast-moving claim against
current documentation** before relying on it. Never invent flags, fields, or benchmark numbers — confirm them.

---

## 12. Canonical references

(Verify URLs/versions — these are authoritative starting points.)

- OWASP Top 10 for LLM Applications — https://genai.owasp.org/
- OWASP Machine Learning Security Top 10 — https://owasp.org/www-project-machine-learning-security-top-10/
- MITRE ATLAS (adversarial threat landscape for AI systems) — https://atlas.mitre.org/
- NIST AI Risk Management Framework — https://www.nist.gov/itl/ai-risk-management-framework
- GKE Sandbox (gVisor) — https://cloud.google.com/kubernetes-engine/docs/concepts/sandbox-pods
- gVisor project — https://gvisor.dev/
- GKE Security Posture — https://cloud.google.com/kubernetes-engine/docs/concepts/about-security-posture-dashboard
- Binary Authorization — https://cloud.google.com/binary-authorization/docs
- Workload Identity Federation for GKE — https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity
- Confidential GKE Nodes — https://cloud.google.com/kubernetes-engine/docs/how-to/confidential-gke-nodes
- Kubernetes Pod Security Admission — https://kubernetes.io/docs/concepts/security/pod-security-admission/
- Kubernetes NetworkPolicy — https://kubernetes.io/docs/concepts/services-networking/network-policies/
- Sigstore / cosign — https://www.sigstore.dev/
- SLSA — https://slsa.dev/
- Falco — https://falco.org/
- Kyverno — https://kyverno.io/ · OPA Gatekeeper — https://open-policy-agent.github.io/gatekeeper/
- safetensors — https://github.com/huggingface/safetensors
- NeMo Guardrails — https://github.com/NVIDIA/NeMo-Guardrails · Guardrails-AI — https://github.com/guardrails-ai/guardrails
- Llama Guard — https://www.llama.com/ (model card; verify current version)
