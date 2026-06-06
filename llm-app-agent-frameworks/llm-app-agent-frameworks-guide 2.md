# LLM Applications & Agent Frameworks — Deep Reference

This is the full ruleset for designing, building, and shipping production LLM applications and agentic
systems. It is framework-aware but not framework-bound: the patterns below outlive any specific SDK.
The ecosystem moves fast (it is 2026) — **verify every framework/API surface and model name against
current docs before you rely on it.** Where this guide names a class, method, or config key, treat it
as illustrative of the *shape* of the API, not a guaranteed signature.

---

## 1. Mental model: an agent is a control-flow graph over an LLM

Strip away the marketing and every "agent" is the same loop:

```
state  ──► build prompt (system + history + tools + context)
       ──► call model
       ──► parse output: final answer? tool call? handoff? plan step?
       ──► if tool call: execute tool, append observation to state, loop
       ──► if final: return
```

The useful unit of thought is **control flow as a graph**: nodes are units of work (an LLM call, a
tool call, a router, a sub-agent), edges are transitions, and there is a shared **state** object that
flows through. The "agent-ness" is just: *the model decides which edge to take next.* The more you let
the model decide, the more autonomous and the less predictable the system. Production engineering is
mostly about **putting bounds back on that autonomy** — step limits, allowed-tool sets, validated
edges, and human checkpoints — without losing the flexibility that made an agent worth building.

Three levels of "agency", pick the least powerful that solves the problem:

1. **Workflow / chain** — you wrote the graph; the LLM fills in nodes. Deterministic edges. Most
   production "AI features" are this. Cheapest, most reliable, easiest to evaluate.
2. **Router + tools** — the LLM picks among a fixed set of branches/tools each turn, but the overall
   structure is bounded. The sweet spot for most agents.
3. **Open-ended agent** — the LLM plans, loops, and decides termination. Powerful, expensive, hard to
   make reliable. Use only when the task genuinely can't be enumerated up front.

> Default bias: **start with the simplest workflow that could work, add autonomy only when an eval
> shows the simpler thing fails.** Autonomy is a cost (latency, tokens, non-determinism, blast radius),
> not a feature.

---

## 2. Core agent design patterns

### ReAct (reason + act)
The model interleaves reasoning ("thought") with tool calls ("action") and observations. Modern
implementations use **native tool/function calling** rather than parsing `Thought:/Action:` text — the
model emits a structured tool call, the runtime executes it, appends the result, and re-prompts.
Prefer native tool-calling over text-scraped ReAct: it's more reliable and the providers train for it.

### Tool / function calling
The model is given JSON-schema tool definitions; it returns a structured call (name + arguments); your
runtime executes and feeds back the result. Rules that matter in production:

- **Tools are an API you are exposing to a probabilistic caller.** Validate arguments server-side
  (don't trust the model's JSON), enforce authz on the *user's* behalf, and make tools idempotent
  where possible.
- **Keep the tool surface small and orthogonal.** 5–15 well-named tools beats 60. Large tool sets
  degrade selection accuracy; if you have many, route/sub-agent so each agent sees a focused subset,
  or use dynamic tool filtering.
- **Tool descriptions are prompt.** Name, description, and parameter docs are how the model decides;
  write them like you'd write docs for a junior engineer. Include when *not* to use the tool.
- **Return structured, terse observations.** Trim tool output to what the model needs; raw 50 KB JSON
  blobs blow the context window and the budget.
- **Parallel tool calls:** most current models can request several tools in one turn. Execute
  independent ones concurrently; preserve call IDs when returning results.

### Planning & decomposition
For multi-step tasks, have the model produce an explicit plan (list of subtasks) before executing.
Plan-and-execute reduces wandering and gives you an artifact to evaluate and to checkpoint against.
Two variants: **plan-once-then-execute** (cheaper, brittle to surprises) and **re-plan after each
step** (robust, more expensive). Bound the number of re-plans.

### Reflection / critique
The model (or a second "critic" model/prompt) reviews its own output against criteria and revises.
Effective for code, structured writing, and math. Caveats: self-critique has diminishing returns
(usually 1–2 rounds), can amplify confident-but-wrong outputs, and adds latency/cost. Bound the
reflection loop and prefer an *external* check (tests, a validator, a different model) over pure
self-grading where you can.

### Routing
A cheap/fast classifier (small model or rules) dispatches the request to the right specialized
handler/agent/prompt. Routing is the highest-ROI pattern: it lets you use a small model for triage and
the big model only where needed, and keeps each downstream prompt focused. Make the router's output a
constrained enum and have a default/fallback branch.

### Multi-agent orchestration
Two dominant topologies:

- **Supervisor / worker (orchestrator–subagent):** a coordinator agent decomposes work and delegates
  to specialist workers (each with its own tools/prompt/context), then synthesizes. Best when subtasks
  are separable and you want isolation of tools/context per specialty.
- **Handoffs (peer-to-peer):** control is transferred between agents (e.g. triage → billing → refund);
  whoever holds the conversation owns it until they hand off. Good for stateful, role-switching
  conversations.

Multi-agent costs: more tokens (context duplicated across agents), harder debugging, more failure
modes. **Don't reach for multi-agent until a single well-tooled agent demonstrably can't do it.** When
you do, give each agent the *minimum* context and tools it needs, and make handoffs explicit and
logged.

### Memory
- **Short-term (working) memory** = the conversation/state in the context window. Manage it: summarize
  or trim old turns, keep a running scratchpad, and protect the system prompt + task framing from being
  evicted.
- **Long-term memory** = persisted facts/preferences/episodes outside the window, retrieved on demand
  (vector store, KV store, or a memory service). Distinguish *semantic* (facts), *episodic* (past
  interactions), and *procedural* (learned how-to) memory. Write deliberately (don't store everything),
  retrieve selectively, and treat retrieved memory as untrusted input. RAG depth → [[rag-vector-databases]].

### Structured output / constrained decoding
When you need machine-parseable output, don't hope — **constrain**. Options, strongest first:
1. **Native structured output / JSON mode** with a supplied JSON Schema (most providers support this).
2. **Grammar / constrained decoding** on self-hosted engines (e.g. vLLM/SGLang with outlines/xgrammar)
   that guarantees the output matches a grammar or schema → [[serving-frameworks]].
3. **Tool calling** as a structured-output channel (define a single "respond" tool with your schema).
4. **Parse-and-retry** with a validator (e.g. a Pydantic model) as a last resort.

Always validate the parsed object against a schema before using it. Reserve a retry budget for
malformed output and log every parse failure — it's an eval signal.

---

## 3. Frameworks — what each is and when to reach for it

| Framework | Shape | State model | Reach for it when |
|---|---|---|---|
| **Google ADK** (Agent Development Kit) | Opinionated agent SDK + runtime; agents, tools, sessions, runners; integrates with the broader Google/Gemini + Vertex stack and deploys to containers/Cloud Run/GKE | Sessions + state services | You want a batteries-included agent framework with first-class deployment, sessions, and Gemini/Vertex integration |
| **LangGraph** | Low-level **graph** runtime: you define nodes + edges + typed shared state; supports cycles, branching, checkpointing, human-in-the-loop, durable execution | Explicit, persisted (checkpointers) | You need precise control over control flow, durable/resumable runs, and stateful multi-agent graphs |
| **LangChain** | Higher-level component/integration library (models, prompts, tools, retrievers, chains) atop the same ecosystem | Lighter; pairs with LangGraph for stateful flows | You want broad ready-made integrations and standard chains; use LangGraph for the agent loop |
| **LlamaIndex** | Data/RAG-centric: ingestion, indexing, retrievers, query engines, plus agent/workflow layers | Workflow + context objects | The app is retrieval/data-heavy; you want strong RAG primitives with agents bolted on |
| **CrewAI** | Role-based multi-agent ("crew" of agents with roles/goals/tasks) | Task/crew state | You want quick role-based multi-agent orchestration with minimal wiring |
| **AutoGen** | Conversational multi-agent framework (agents that message each other); event-driven core | Conversation/event state | Research-y multi-agent conversations, code-exec agents, experimentation |
| **Pydantic-AI** | Type-first, minimal agent library built around Pydantic models for structured I/O and deps | Lightweight, app-owned | You want type-safe, dependency-injected agents without a heavy runtime |
| **Haystack** | Pipeline framework (nodes/components) historically RAG/search-focused, now agent-capable | Pipeline state | Search/RAG pipelines and you like the component-pipeline model |

How they differ along the axes that matter:

- **Abstraction level:** LangGraph / Pydantic-AI are low-level (you own the loop); ADK / CrewAI /
  AutoGen / LangChain chains are higher-level (the framework owns more of the loop). Lower level = more
  control + more code; higher level = faster start + more magic to debug.
- **Statefulness:** LangGraph and ADK have first-class persisted state/sessions and resumability;
  others lean on you to persist. For long-running, resumable, or human-in-the-loop agents, prefer a
  framework with real checkpointing.
- **Control vs. autonomy:** graph frameworks make edges explicit (good for production reliability);
  conversational/role frameworks hide control flow inside agent messaging (faster to prototype, harder
  to bound).

> **Framework selection rule:** choose for the *control model* and *state model* you need, plus the
> integrations you'd otherwise rebuild. Don't let a framework own your prompts and business logic so
> deeply that you can't swap models or escape it. Keep your domain logic in plain functions; let the
> framework orchestrate. **Re-verify each framework's current API surface — these libraries break and
> rename across minor versions.**

---

## 4. MCP — Model Context Protocol

**MCP is the open standard for connecting tools, data, and context to LLM applications** — a USB-C port
for models. It decouples *who provides a capability* (an MCP **server**) from *who consumes it* (an MCP
**client**, embedded in your agent/host app). Instead of hand-wiring every integration into every
agent, you run/point at MCP servers and any MCP-aware client can use them.

### Architecture
- **Host** — the application the user interacts with (your agent app, an IDE, a chat client).
- **Client** — lives in the host; maintains a 1:1 connection to one server, handles the protocol.
- **Server** — exposes capabilities. Three primitives:
  - **Tools** — model-invocable functions (actions/side-effects). The agent calls these.
  - **Resources** — readable data/context (files, records, docs) the host can pull in.
  - **Prompts** — reusable templated prompts/workflows the user or host can invoke.
- **Transports** — **stdio** (local subprocess; simplest, great for local/dev and co-located tools) and
  **streamable HTTP** (remote servers; supports streaming). The protocol is JSON-RPC based.

### When to use MCP
- You want to **reuse the same tool/integration across multiple agents or multiple agent frameworks**
  without recoding it per SDK.
- You want a **clean trust/security boundary**: the server owns credentials and the actual integration;
  the agent only sees the declared tools/resources.
- You're consuming third-party or off-the-shelf connectors.

When *not* to: for a single agent with a couple of in-process Python functions, a plain tool is
simpler — don't add a protocol hop you don't need.

### Security (this is the part people get wrong)
- **Treat every MCP server as untrusted code and every tool result as untrusted input.** Tool
  descriptions and returned content can carry **prompt-injection** payloads ("ignore previous
  instructions…"). Don't let raw tool/resource output silently become trusted instructions.
- **Pin and vet servers.** Prefer first-party/known servers; review what a server can reach. A
  malicious or compromised server can exfiltrate context or perform actions.
- **Scope credentials at the server**, least-privilege, per-user where possible. Don't hand a broad
  token to a remote server you don't control.
- **Confirm or sandbox destructive tools** (writes, money movement, code exec). Require human approval
  for high-blast-radius actions. Sandboxing → [[ai-security-on-gke]].
- Frameworks (ADK, LangGraph/LangChain, LlamaIndex, Pydantic-AI, etc.) increasingly ship MCP client
  adapters so MCP tools appear as native tools — **verify the adapter and version against current
  docs**, as this surface is young and moving.

---

## 5. Connecting to models

### OpenAI-compatible APIs are the lingua franca
Most providers and self-hosted engines (vLLM, SGLang, etc.) expose an **OpenAI-compatible** `chat
completions` / `responses`-style endpoint. Standardize your app on that interface and you can swap
between hosted and self-hosted by changing a base URL + key. Keep a thin **provider abstraction** so
model choice is config, not code.

- **Self-hosted serving** (you own the GPUs/throughput/cost): vLLM, SGLang, TensorRT-LLM, Triton, Ray
  Serve, KServe → [[serving-frameworks]]. Use when you need data control, custom/fine-tuned models,
  predictable cost at volume, or constrained decoding/grammars.
- **Inference routing / gateway:** put a gateway in front to do model-aware load balancing, fallback,
  and traffic splitting across replicas/models → [[gke-inference-gateway]]. On Kubernetes broadly →
  [[aiml-on-kubernetes]].

### Cost / latency / streaming
- **Stream** tokens (SSE) for any user-facing turn — it slashes perceived latency. For non-user-facing
  agent steps, streaming mainly helps you cut off bad generations early.
- **Right-size the model per node.** Use a small/cheap model for routing, classification, extraction,
  and tool-arg formatting; reserve the frontier model for hard reasoning. This is the single biggest
  cost lever.
- **Cache aggressively** (see §8): prompt/prefix caching on the provider side, plus your own
  response/embedding cache for deterministic sub-calls.
- **Watch the token bill compounding:** every agent turn re-sends growing history × every tool result ×
  every reflection round. Loops multiply tokens. Budget and cap (§7, §9).

---

## 6. Deploying agents on Kubernetes / GKE

Agents are usually a stateless-ish HTTP service in front of stateful sessions and slow tool calls.
Design for that shape. K8s fundamentals → [[kubernetes-expert]] / [[aiml-on-kubernetes]].

- **Containerize** the agent service; keep the image lean. Model weights are *not* in the agent image —
  the model is an API (hosted or a serving deployment).
- **Stateless service, externalized session/state.** Put conversation/session state in a store (Redis,
  Postgres, a session service, or your framework's checkpointer backend) so any replica can serve any
  turn and you can scale horizontally and survive pod restarts. LangGraph checkpointers / ADK session
  services are built for this.
- **Long-running & concurrency:** agent turns can take seconds to minutes (tool calls, multi-step
  loops). Decide per workload:
  - **Synchronous request/response** with generous timeouts for short interactive turns.
  - **Async + queue + worker** (e.g. a task queue/Pub-Sub + worker pods) for long autonomous runs;
    return a job id, stream/poll progress. This decouples request lifetime from agent runtime and lets
    you retry and bound work.
  - Each in-flight agent run holds memory + open tool connections — concurrency is bounded by that, not
    just CPU. Size requests/limits accordingly and cap concurrent runs per pod.
- **Scaling:** scale on a real signal — in-flight runs, queue depth, or request concurrency — not CPU%
  (LLM calls are I/O-bound waits). Use KEDA on queue depth for the worker pattern; HPA on a custom
  concurrency metric for the sync pattern → [[autoscaling-kubernetes]]. Keep startup fast for
  scale-to-zero/burst.
- **Secrets:** model API keys, tool credentials, MCP-server creds — via a secrets manager / mounted
  secrets, never baked into images or prompts. Rotate. Least-privilege per tool.
- **Sandbox untrusted tool/code execution.** If the agent runs generated code or calls tools that touch
  the network/filesystem, isolate it: **gVisor (`runsc`) runtime class**, a dedicated sandbox
  pod/microVM, no node credentials, egress-restricted, ephemeral. Never run model-generated code in the
  agent's own process or with cluster permissions → [[ai-security-on-gke]].

---

## 7. RAG integration (brief — depth in [[rag-vector-databases]])

RAG = retrieve relevant context, inject it into the prompt, generate grounded in it. In agentic systems
RAG usually shows up as a **retrieval tool** the agent calls (agentic RAG) rather than a fixed
pre-fetch. Essentials:

- Chunk + embed + index your corpus; retrieve top-k (often hybrid: vector + keyword) and optionally
  re-rank. Keep retrieved spans small and cite their source.
- **Ground and attribute:** instruct the model to answer *from* the context and to say when it can't.
  Carry source ids through so you can show citations and evaluate grounding.
- **Retrieved content is untrusted input** — same prompt-injection risk as tool output. Don't let a
  retrieved document issue instructions.
- Everything about chunking strategy, embeddings, vector DBs, hybrid search, and re-ranking lives in
  [[rag-vector-databases]] — defer there.

---

## 8. Production concerns

### Evaluation (you cannot ship an agent you can't measure)
- **Offline evals**: a versioned dataset of inputs + expected behavior. Evaluate at multiple levels —
  final-answer quality, per-step tool-selection correctness, retrieval quality, and trajectory (did it
  take a sane path). Run on every prompt/model/framework change in CI. **No evals = no production.**
- **Online evals**: sample real traffic; track success/abandonment, user feedback (thumbs), guardrail
  hits, cost/latency per session, loop/step counts.
- **LLM-as-judge**: use a model to grade outputs against a rubric when there's no exact-match answer.
  Make it cheap, give it a clear rubric, calibrate it against human labels periodically, and beware its
  biases (position, verbosity, self-preference). Use pairwise comparison where you can.
- **Component evals**: test routers, extractors, and tools in isolation with deterministic cases.

### Guardrails
Input and output filters around the model: block/redact PII and secrets, detect prompt injection and
jailbreaks, constrain outputs to policy, and gate high-risk tool calls behind approval. Guardrails are
defense-in-depth, not a substitute for least-privilege tools and sandboxing → [[ai-security-on-gke]].

### Observability / tracing
- **Trace every run** as a tree of spans: each LLM call, tool call, retrieval, and sub-agent is a span
  with inputs, outputs, token counts, latency, and cost. This is non-negotiable for debugging agents —
  you cannot reason about a loop you can't see.
- **Use OpenTelemetry GenAI semantic conventions** for spans/metrics (e.g. `gen_ai.*` attributes —
  model, token usage, operation) so traces are portable across tooling. Frameworks and tracing
  platforms increasingly emit these — **verify the current convention attribute names against the
  OpenTelemetry spec**, as GenAI conventions are still evolving.
- Capture and version prompts with each trace so a regression is reproducible.

### Prompt & version management
- **Prompts are code**: version them, review changes, and tie each deployed version to its eval
  results. Externalize prompts from code so you can iterate/roll back without redeploying, but keep
  them under source control.
- Pin model versions explicitly; a silent model upgrade can shift behavior. Re-run evals on any
  model/prompt bump.

### Cost & loop-bounding
- **Every loop gets a hard cap**: max steps/turns, max tool calls, max tokens, wall-clock timeout, and
  a max-cost budget per run. When a cap trips, fail gracefully (return best-effort + flag), don't spin.
- Detect and break **degenerate loops** (same tool + same args repeatedly, oscillation) — track a small
  history and abort on repetition.

### Determinism & caching
- LLMs are non-deterministic even at temperature 0 (kernel/batching effects). **Don't depend on
  bit-exact reproducibility.** For sub-tasks that should be stable, cache by a hash of the exact input
  and reuse.
- **Prompt/prefix caching** (provider- or engine-side) cuts cost/latency when a long stable prefix
  (system prompt, tool defs, retrieved context) is reused — order your prompt to maximize the cacheable
  prefix (stable stuff first, variable stuff last).
- Cache embeddings and deterministic tool results. Make caching observable (hit rate is a cost metric).

---

## 9. Anti-patterns (these cause the production incidents)

- **Unbounded agent loops → runaway cost / hangs.** No step cap, no token budget, no timeout, no
  loop-detection. A single stuck session can burn thousands of dollars. *Always* bound every loop.
- **No evals.** Shipping prompt/model changes on vibes. You will regress silently and find out from
  users. Build a dataset before you build the agent.
- **No tool sandboxing.** Running model-generated code or network/filesystem tools in-process or with
  cluster/cloud credentials. One injection and the agent acts as you. Sandbox (gVisor/microVM),
  least-privilege, egress-restrict, human-gate destructive actions.
- **Prompt spaghetti.** Giant, unversioned, string-concatenated prompts with logic buried in them. Make
  prompts data, version them, keep business logic in code.
- **Hidden non-determinism.** Depending on exact output strings, unpinned models, or "it worked once."
  Constrain output, validate, pin versions, and design for variance.
- **Trusting tool/RAG/MCP output blindly.** Treating retrieved or tool-returned text as trusted
  instructions → prompt injection, data exfiltration. All external content is untrusted input.
- **Over-agentifying.** Multi-agent swarm for a job a single chain would do — more cost, more failure
  modes, harder evals. Use the least autonomy that works.
- **Giant tool sets / overlapping tools.** Degrades selection; route or split per sub-agent.
- **Leaking secrets into prompts/logs/traces.** Redact before logging; never put live credentials in
  context.
- **No observability.** Debugging an agent without per-step traces is hopeless. Trace from day one.

---

## 10. Version awareness

This space changes monthly. Model names, context limits, pricing, structured-output support, MCP spec
revisions and transport details, OpenTelemetry GenAI convention attribute names, and every framework's
API (especially LangChain/LangGraph, ADK, LlamaIndex, CrewAI, AutoGen) all move. **Before relying on
any specific class name, method signature, config key, model id, or limit in this guide, check the
current official docs.** The *patterns* here are durable; the *surfaces* are not.

---

## 11. Canonical references (verify currency)

- **Model Context Protocol** — spec & docs: https://modelcontextprotocol.io and
  https://spec.modelcontextprotocol.io
- **Google ADK** — https://google.github.io/adk-docs/
- **LangGraph / LangChain** — https://langchain-ai.github.io/langgraph/ and https://python.langchain.com/
- **LlamaIndex** — https://docs.llamaindex.ai/
- **CrewAI** — https://docs.crewai.com/ · **AutoGen** — https://microsoft.github.io/autogen/ ·
  **Pydantic-AI** — https://ai.pydantic.dev/ · **Haystack** — https://docs.haystack.deepset.ai/
- **OpenTelemetry GenAI semantic conventions** —
  https://opentelemetry.io/docs/specs/semconv/gen-ai/
- **gVisor (runsc sandbox)** — https://gvisor.dev/docs/
- Anthropic, "Building effective agents" — https://www.anthropic.com/research/building-effective-agents
- ReAct paper — https://arxiv.org/abs/2210.03629 · Reflexion — https://arxiv.org/abs/2303.11366
