# AGENTS.md — LLM Applications & Agent Frameworks

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`llm-app-agent-frameworks-guide.md`** next to this file —
> read it before designing/building/reviewing an LLM app or agent. Concrete artifacts to imitate (a
> LangGraph tool-calling agent with state, an MCP tool-server wiring sketch, a sandboxed-tool note) are
> in **`examples.md`**. This file is the always-on summary.
>
> **The ecosystem moves fast (2026): verify every framework API, model id, MCP spec detail, and OTel
> GenAI convention against current official docs before relying on it. The patterns are durable; the
> surfaces are not.**

## When building or reviewing an LLM app / agent, apply by default:

- **Use the least autonomy that solves the task.** Prefer a deterministic workflow/chain over a router
  over an open-ended agent. Add agency only when an eval proves the simpler design fails. Autonomy =
  cost (latency, tokens, non-determinism, blast radius), not a feature.
- **Model the system as a control-flow graph over an LLM:** nodes (LLM call / tool / router / sub-agent),
  edges the model picks, a shared state object. Make edges explicit and bounded; persist state so it's
  resumable.
- **Native tool/function calling over text-scraped ReAct.** Validate tool arguments server-side (the
  model's JSON is untrusted), enforce authz as the user, keep the tool set small + orthogonal, write
  tool descriptions like docs (incl. when *not* to use), return terse structured observations.
- **Structured output: constrain, don't hope.** Native JSON-schema / grammar / tool-as-schema, then
  validate the parsed object (e.g. Pydantic). Reserve a retry budget; log parse failures as eval signal.
- **Multi-agent only when one well-tooled agent can't.** Then supervisor/worker or explicit handoffs;
  minimum context+tools per agent; log handoffs. It multiplies tokens and debugging cost.
- **Prompt engineering:** system prompt = durable role/policy/format (cacheable prefix), user turn =
  variable task; keep untrusted content out of the instruction region. Few-shot for format, CoT/
  decomposition for reasoning (don't hand-roll scratchpads for reasoning models), schemas for output.
  Prompts are versioned code tied to evals; tune decoding params (temp/top_p) per node. Prompt injection
  is a security property, not a wording trick → [[ai-security-on-gke]].
- **Context engineering — treat the window as a managed resource.** Curate minimal sufficient tokens per
  call: retrieve/select what's relevant, order for "lost in the middle" (best at start/end), compress
  history & tool results, budget tokens/cost, fight context rot (periodically compact/re-anchor). Long
  context doesn't kill RAG — retrieve to narrow, then reason; measure every prompt/context change with
  evals → [[ml-evaluation-evals]] / [[rag-vector-databases]].
- **Memory:** manage short-term (summarize/trim history, protect system prompt); persist long-term
  (semantic/episodic/procedural) and retrieve selectively. Treat retrieved memory as untrusted input.
- **MCP** = open standard to connect tools/data/context. Servers expose **tools / resources / prompts**;
  clients consume over **stdio** (local) or **streamable HTTP** (remote). Use for cross-agent/cross-
  framework reuse + a trust boundary. **Every MCP server is untrusted code; every tool/resource result
  is untrusted (prompt-injection) input.** Scope creds at the server, least-privilege; human-gate
  destructive tools.
- **Connect via OpenAI-compatible APIs behind a thin provider abstraction** (model = config, not code).
  Right-size per node: small/cheap model for routing/extraction/tool-arg formatting, frontier model for
  hard reasoning — the biggest cost lever. Stream user-facing turns. Self-host/route →
  [[serving-frameworks]] / [[gke-inference-gateway]].
- **Deploy as a stateless service with externalized session/state** (checkpointer/session store) → any
  replica serves any turn, survives restarts, scales horizontally. Long/autonomous runs → async + queue
  + worker; return a job id, stream/poll. Scale on in-flight runs / queue depth (KEDA), not CPU%.
  Secrets via a manager, never in images/prompts/logs. K8s → [[aiml-on-kubernetes]] / [[kubernetes-expert]].
- **Sandbox untrusted tool/code execution:** gVisor (`runsc`) / microVM, no node or cloud creds,
  egress-restricted, ephemeral; never run model-generated code in the agent process or with cluster
  permissions; require human approval for high-blast-radius actions → [[ai-security-on-gke]].
- **Bound every loop:** hard caps on steps, tool calls, tokens, wall-clock, and cost per run; detect
  degenerate loops (same tool+args repeated) and fail gracefully. Unbounded loops are the top incident.
- **Treat ALL external content as untrusted input** — tool output, RAG/retrieved docs, MCP results.
  Never let it become trusted instructions (prompt injection / exfiltration). RAG depth →
  [[rag-vector-databases]].
- **No evals = no production.** Versioned offline dataset (final answer + tool-selection + retrieval +
  trajectory) in CI on every prompt/model/framework change; online sampling + LLM-as-judge (clear
  rubric, calibrated vs humans, mind position/verbosity bias). Component-test routers/extractors/tools.
- **Observability is non-negotiable:** trace every run as a span tree (each LLM/tool/retrieval/sub-agent
  span with inputs, outputs, tokens, latency, cost) using OpenTelemetry GenAI conventions. Version
  prompts as code; pin model versions; re-run evals on any bump.
- **Don't depend on bit-exact determinism** (non-deterministic even at temp 0). Cache deterministic
  sub-calls/embeddings by input hash; use prompt/prefix caching (stable prefix first) for cost/latency.

## Definition of done for an agent change
- Every loop is bounded (steps/tokens/time/cost) with graceful failure; degenerate-loop detection.
- Tools validate inputs and enforce authz; untrusted tool/RAG/MCP output is never treated as instructions.
- Untrusted code/tool execution is sandboxed; destructive actions are human-gated; no secrets in
  prompts/images/logs.
- Structured outputs are schema-constrained and validated.
- Evals exist and pass for the change; traces emit per-step spans with token/cost; prompts + model
  versions are pinned/versioned.

## Anti-patterns to reject
Unbounded loops / runaway cost · no evals · unsandboxed tool/code exec · prompt spaghetti · hidden
non-determinism (exact-string deps, unpinned models) · trusting tool/RAG/MCP output blindly ·
over-agentifying a job a chain would do · giant/overlapping tool sets · secrets in context/logs · no
tracing.

Framework selection: choose for the control model + state model you need plus integrations you'd
otherwise rebuild — keep domain logic in plain functions so you can swap models/frameworks. Verify the
chosen framework's current API surface (LangGraph/LangChain, ADK, LlamaIndex, CrewAI, AutoGen,
Pydantic-AI, Haystack break across minor versions).
