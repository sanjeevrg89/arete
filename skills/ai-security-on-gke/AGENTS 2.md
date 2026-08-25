# AGENTS.md — AI/LLM Security on GKE (defensive)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`ai-security-on-gke-guide.md`** next to this file — read it before
> threat-modeling or hardening an AI/LLM workload, and apply it. Concrete manifests to imitate (gVisor
> sandbox Pod, default-deny + egress NetworkPolicy, Workload Identity least-privilege) are in **`examples.md`**.
> This is a **defensive** skill: protect AI workloads, never attack. This file is the always-on summary.
>
> Proprietary / fast-moving items (managed-product features, exact API surfaces, model versions) are
> **verify against current docs**. Never fabricate APIs, flags, or benchmarks — confirm them.

## When working on AI/LLM workloads (inference, RAG, agents) on K8s/GKE, apply by default:

- **Trust nothing the model touches.** The prompt, the model output, retrieved/tool/web content, and model
  weights are all untrusted. The model is not a security boundary; controls live around it.
- **Filter both directions, at the gateway.** Input guardrail (prompt-injection/jailbreak/PII) before the
  model; output guardrail (toxicity/PII/secret leak) and **tool-call validation** after. Re-screen RAG/tool
  content before it re-enters context. Options: Model Armor (managed), Llama Guard, NeMo Guardrails,
  Guardrails-AI. Fail closed for irreversible actions. No secrets in system prompts.
- **Indirect prompt injection is the top agentic threat.** Malicious instructions ride in via RAG docs, web
  pages, emails, tool results. Delimit untrusted content as data; never concatenate it into privileged
  instructions unmarked; constrain downstream agency.
- **Excessive agency is a design flaw.** Least-privilege tools, per-action authz, human approval for
  irreversible/high-blast-radius actions, step/loop caps. Split identities — not one all-powerful agent SA.
- **Sandbox all untrusted code/tool execution.** gVisor (GKE Sandbox) or Kata/microVM. Pod hardening:
  non-root, `readOnlyRootFilesystem: true`, `capabilities.drop: ["ALL"]`, `seccompProfile: RuntimeDefault`,
  `automountServiceAccountToken: false`, resource + step/time caps, ephemeral per session, **no egress**.
- **Egress control is the exfiltration backstop.** Default-deny `NetworkPolicy` per namespace; allow only DNS
  + required destinations. Private clusters; egress proxy/allowlist for internet-bound tools.
- **No static keys.** Workload Identity Federation → short-lived scoped credentials. Per-workload
  least-privilege identities mapping IAM ↔ RBAC. Secrets in Secret Manager; never logged or in prompts.
- **Verify provenance.** Prefer `safetensors` over pickle/`torch.load` (pickle executes code on load); pickle
  only from a trusted, scanned source in a credential-less, no-egress sandbox. Pin images/weights by digest.
  Sign & verify images (Binary Authorization / Sigstore); SLSA provenance; scan artifacts; dataset integrity.
- **Admission + runtime.** PSA `restricted` by default; Gatekeeper/Kyverno (require signed images, sandbox
  runtimeClass, resource limits, default-deny NetworkPolicy; forbid privileged/hostNetwork/hostPath). Runtime
  detection via GKE Security Posture / Container Threat Detection and/or Falco; route findings to SIEM.
- **Protect against Model DoS & theft.** Rate-limit, cap tokens/context/steps. Treat weights as crown jewels:
  access control, egress limits, Confidential GKE for sensitive weights/data; tenant isolation.

## Anti-patterns (reject these)
Trusting model/tool output (exec/SQL/HTML/tool-call without validation) · no egress controls · untrusted code
unsandboxed · static SA JSON keys · loading pickle weights from untrusted sources · input-only filtering ·
one over-permissioned agent identity · "tell the model not to" as a control · mutable image/model tags in prod.

## Map every threat to a control
Use the OWASP LLM Top 10 → control table and the defense-in-depth mapping in `ai-security-on-gke-guide.md`.
Definition of done: a named control at every layer (gateway, agent, sandbox, admission, runtime, supply chain,
identity, network, data), no single point of trust.

## Related skills
`[[gke-inference-gateway]]` · `[[llm-app-agent-frameworks]]` · `[[aiml-on-kubernetes]]` · `[[gke-master]]` ·
`[[kubernetes-expert]]`
