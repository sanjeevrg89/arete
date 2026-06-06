---
name: llm-app-agent-frameworks
description: Expert guidance for building and shipping production LLM applications and agentic systems.
  Use when designing or reviewing agents — ReAct, tool/function calling, planning, reflection, routing,
  multi-agent (supervisor/worker, handoffs), short/long-term memory, structured output / constrained
  decoding, control-flow-as-graph. Covers prompt engineering and context engineering (system/user
  prompts, few-shot, chain-of-thought, output schemas, prompt templating/versioning, decoding params,
  context-window budgeting, retrieval/ordering/compression, lost-in-the-middle, context rot,
  long-context-vs-RAG, prompt injection). Covers framework choice (Google ADK, LangChain/LangGraph,
  LlamaIndex, CrewAI, AutoGen, Pydantic-AI, Haystack), MCP (Model Context Protocol) servers/clients/
  transports/tools, connecting to OpenAI-compatible and self-hosted models, deploying agents on
  Kubernetes/GKE (stateless vs session, queueing, scaling, sandboxed tool execution), RAG integration,
  and production concerns: evals (LLM-as-judge), guardrails, OpenTelemetry GenAI tracing, prompt/version
  management, cost & loop-bounding, caching. Triggers on agent loops, tool-calling code, langgraph/
  langchain/adk/llama_index/crewai/autogen imports, mcp servers, and "build/ship an agent" tasks.
---

# LLM Applications & Agent Frameworks

Apply the judgment of an engineer who has shipped production agentic systems: the right level of
autonomy for the job, bounded loops, sandboxed tools, real evals, and full tracing. **The single most
important instinct: an agent is a control-flow graph over an LLM — use the least autonomy that solves
the problem, and put hard bounds back on everything the model gets to decide.**

## How to use this skill

1. **Read `llm-app-agent-frameworks-guide.md`** in this directory — the full reference (patterns,
   framework selection, MCP, deployment, production concerns, anti-patterns). Apply it to the task.
2. For concrete artifacts to imitate (a LangGraph tool-calling agent with state, an MCP tool-server
   wiring sketch, and a sandboxed-tool-execution note), read **`examples.md`**.
3. Match the surrounding codebase/cluster conventions and the chosen framework's current API; apply the
   correctness/safety rules (loop bounds, sandboxing, evals, untrusted-input handling) regardless.
   **The ecosystem moves fast — verify framework APIs, model ids, MCP spec, and OTel GenAI conventions
   against current docs before relying on them.**

## Essentials (full detail in `llm-app-agent-frameworks-guide.md`)

- **Least autonomy that works.** Workflow/chain → router+tools → open-ended agent, in that order of
  preference. Add agency only when an eval shows the simpler design fails. Autonomy is a cost.
- **Control flow is a graph** over shared state: nodes (LLM call, tool, router, sub-agent), edges
  (transitions the model picks), a state object that flows through. Make edges explicit and bounded.
- **Tools are an API exposed to a probabilistic caller.** Validate args server-side, enforce authz,
  keep the set small/orthogonal, write descriptions like docs, return terse structured observations.
- **Multi-agent only when a single well-tooled agent can't.** Then prefer supervisor/worker or explicit
  handoffs; give each agent minimum context + tools; log handoffs. It costs tokens and debuggability.
- **Constrain structured output** (native JSON-schema / grammar / tool-as-schema), then validate the
  parsed object. Never depend on exact output strings or bit-exact determinism (true even at temp 0).
- **Prompt engineering as a discipline.** System prompt = durable role/policy/format (cacheable prefix);
  user turn = variable task; keep untrusted tool/RAG content out of the instruction region. Few-shot for
  format, CoT/decomposition for reasoning (let reasoning models reason — don't hand-roll scratchpads),
  schemas for output. Prompts are versioned code tied to evals; tune decoding params (temp/top_p) per
  node. Prompt injection is a security property, not a wording trick → [[ai-security-on-gke]].
- **Context engineering — the window is a managed resource.** Curate the minimal sufficient tokens per
  call: retrieve/select what's relevant, order it for "lost in the middle" (best at start/end), compress
  history & tool results, budget tokens/cost, and fight context rot (periodically compact/re-anchor).
  Long context doesn't kill RAG — retrieve to narrow, then reason; measure both with evals.
- **MCP** is the open standard to connect tools/data/context: servers expose tools/resources/prompts,
  clients consume over stdio or streamable HTTP. Use it for cross-agent reuse and a trust boundary.
  Treat every MCP server as untrusted code and every tool/resource result as untrusted (injection) input.
- **Connect via OpenAI-compatible APIs** behind a thin provider abstraction so model choice is config.
  Right-size the model per node (small for routing/extraction, frontier for hard reasoning); stream
  user-facing turns. Self-hosting/routing → [[serving-frameworks]] / [[gke-inference-gateway]].
- **Deploy as a stateless service with externalized session/state** (checkpointer/session store) so any
  replica serves any turn. Long/autonomous runs → async + queue + worker; scale on in-flight runs /
  queue depth, not CPU. Secrets via a manager, never in images/prompts.
- **Sandbox untrusted tool/code execution** — gVisor (`runsc`)/microVM, no node creds, egress-restricted,
  ephemeral; human-gate destructive actions. Never run model-generated code in-process → [[ai-security-on-gke]].
- **Bound every loop**: max steps, tool calls, tokens, wall-clock, and cost budget; detect degenerate
  loops (repeated tool+args) and fail gracefully. Unbounded loops are the #1 cost/incident cause.
- **No evals = no production.** Offline dataset (final answer, tool-selection, retrieval, trajectory) in
  CI on every prompt/model change; online sampling + LLM-as-judge (calibrated against humans).
- **Trace every run** as a span tree (LLM/tool/retrieval/sub-agent) with tokens, latency, cost; use
  OpenTelemetry GenAI conventions. Version prompts as code and pin model versions.

## Related skills

- `[[rag-vector-databases]]` — retrieval, chunking, embeddings, vector DBs, hybrid search, re-ranking
  (RAG depth; this skill only sketches integration). The core context-engineering lever.
- `[[ml-evaluation-evals]]` — eval discipline (offline datasets, LLM-as-judge, CI gating) for measuring
  prompt and context-engineering changes; pair with every prompt/context strategy change.
- `[[ai-security-on-gke]]` — guardrails, sandboxing untrusted tool/code execution, agent threat model.
- `[[serving-frameworks]]` — self-hosting models (vLLM/SGLang/Triton/KServe) + constrained decoding.
- `[[gke-inference-gateway]]` — model-aware routing/load-balancing/fallback in front of model replicas.
- `[[aiml-on-kubernetes]]` — running AI/ML (incl. agentic) workloads on K8s/GKE (umbrella).
- `[[kubernetes-expert]]` — general K8s deployment/scaling/secrets the agent service runs on.
