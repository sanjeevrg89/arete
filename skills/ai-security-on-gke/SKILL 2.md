---
name: ai-security-on-gke
description: Defensive, defense-in-depth security for AI/LLM workloads on Kubernetes and GKE — at the bar of a
  security engineer for an AI platform. Use when threat-modeling or hardening LLM inference, RAG, or agentic
  apps; designing prompt-injection / jailbreak / PII / toxicity filtering (Model Armor, Llama Guard, NeMo
  Guardrails, Guardrails-AI); sandboxing untrusted tool/code execution (gVisor/GKE Sandbox, Kata/microVMs,
  seccomp, AppArmor); runtime threat detection (GKE Security Posture, Container Threat Detection, Falco);
  admission control & policy (Pod Security Admission, Gatekeeper, Kyverno); image & model supply-chain
  (Binary Authorization, Sigstore/SLSA, safetensors vs pickle, dataset integrity); identity, secrets, and
  egress control (Workload Identity Federation, Secret Manager, NetworkPolicy, private clusters, Confidential
  GKE). Maps the OWASP LLM Top 10 to concrete controls.
---

# AI Security on GKE

Apply the judgment of a security engineer responsible for an AI platform serving untrusted prompts at scale.
This is a **defensive** skill: protect AI/LLM workloads with defense-in-depth — assume the model output, the
retrieved documents, and any tool the agent calls are **all untrusted**, and put a control at every layer.

## How to use this skill

1. **Read `ai-security-on-gke-guide.md`** in this directory — the full reference (threat model → control
   mapping, content/model defense, sandboxing, runtime, supply chain, identity/secrets/network, checklist).
   Apply it to the workload at hand.
2. For concrete, imitate-this manifests (gVisor `RuntimeClass` + sandboxed Pod, default-deny + controlled-
   egress `NetworkPolicy`, Workload Identity least-privilege), read **`examples.md`**.
3. Match the cluster's existing platform conventions; apply the correctness/safety rules regardless. Treat
   proprietary or fast-moving items (managed-product features, API surfaces) as **verify against current
   docs** — the ecosystem moves fast and it is 2026.

## Essentials (full detail in `ai-security-on-gke-guide.md`)

- **Trust nothing the model touches.** Treat LLM output, RAG/tool/web content, and prompts as untrusted
  input. Filter on the way **in** (prompt injection, jailbreak) and on the way **out** (PII/secret leak,
  toxicity, unsafe actions). Place guardrails at the gateway — see `[[gke-inference-gateway]]`.
- **Indirect prompt injection is the dominant threat for RAG/agents.** Malicious instructions ride in via
  retrieved docs, web pages, emails, or tool results — not the user message. Untrusted content must never be
  concatenated into a privileged instruction context unmarked.
- **Sandbox all untrusted code/tool execution.** Code interpreters and agent tools run in gVisor (GKE
  Sandbox) or Kata/microVM isolation, with `seccompProfile: RuntimeDefault`, dropped capabilities,
  read-only rootfs, no host network, resource caps, and **no egress by default**.
- **Excessive agency is a design flaw, not a model flaw.** Constrain what tools an agent can call, scope
  every credential to least privilege, and require human approval for irreversible/high-blast-radius actions.
- **Egress control is your data-exfiltration backstop.** Default-deny `NetworkPolicy`; allow only the
  specific egress an inference/agent Pod needs (DNS, model registry, approved APIs). Assume a compromised
  prompt will try to phone home.
- **No static keys.** Use Workload Identity Federation so Pods get short-lived, scoped credentials; never
  bake service-account JSON keys into images or Secrets. Map IAM ↔ RBAC to least privilege.
- **Verify model & image provenance.** Sign and verify images (Binary Authorization / Sigstore); prefer
  `safetensors` over pickle/`torch.load` (pickle executes arbitrary code on load); scan artifacts; track
  SLSA provenance and dataset integrity.
- **Defense in depth — every layer gets a control.** Admission (Pod Security Admission + Gatekeeper/Kyverno),
  runtime threat detection (GKE Security Posture / Container Threat Detection / Falco), private clusters,
  encryption, and Confidential GKE for sensitive weights/data. See `[[gke-master]]`, `[[kubernetes-expert]]`.
- **Defend against model DoS and theft.** Rate-limit and cap tokens/context per caller; protect weights as a
  crown-jewel asset (access control, egress limits, watermarking where applicable).
- **Anti-patterns:** trusting model/tool output, no egress controls, running untrusted code unsandboxed,
  static SA keys, loading pickle weights from untrusted sources, no output filtering, one big over-permissioned
  agent service account.

## Related skills

- `[[gke-inference-gateway]]` — where to terminate, route, and attach inference guardrails at the gateway.
- `[[llm-app-agent-frameworks]]` — building the agentic apps these controls wrap (tools, memory, planning).
- `[[aiml-on-kubernetes]]` — umbrella for training/inference/agentic AI on K8s & GKE.
- `[[gke-master]]` — cluster-level security: private clusters, Confidential GKE, node pools, networking.
- `[[kubernetes-expert]]` — core K8s security primitives (RBAC, NetworkPolicy, PSA, admission).
