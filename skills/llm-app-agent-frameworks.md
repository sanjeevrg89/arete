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
  and agent reliability / durable execution for long-running agents (durable workflow engines — Temporal,
  AWS Bedrock AgentCore, Restate, DBOS, Inngest — checkpoint/replay/resume, workflow-vs-activity split,
  idempotency, retries/backoff, exactly-once side effects, timeouts/heartbeats, human-in-the-loop
  pause/resume), plus production concerns: evals (LLM-as-judge), guardrails, OpenTelemetry GenAI tracing,
  prompt/version management, cost & loop-bounding, caching. Triggers on agent loops, tool-calling code,
  langgraph/langchain/adk/llama_index/crewai/autogen imports, mcp servers, durable execution, agent
  reliability, long-running/async agents, and "build/ship an agent" tasks.
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
- **Durable execution for long-running agents.** A naive in-process loop loses the whole run on crash and
  can't safely resume — reliability is a distributed-systems problem → [[distributed-systems-fundamentals]].
  Use a durable workflow engine (Temporal/AWS Bedrock AgentCore/Restate/DBOS/Inngest) or a framework
  checkpointer: deterministic **workflow code** (orchestration only) vs. **activities** (all LLM/tool I/O,
  checkpointed and not re-run on replay). Make tool calls **idempotent** (stable idempotency keys), put
  bounded **retries/backoff** + timeouts/heartbeats on activities, get **exactly-once** side effects via
  dedupe. Enables human-in-the-loop pause/resume, durable timers, multi-day flows, checkpoint/resume of
  agent memory. Run stateless workers against a durable backend on K8s → [[kubernetes-expert]]; trace the
  durable history + failure modes → [[ml-observability-monitoring]] / [[ml-evaluation-evals]].
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
  prompt and context-engineering changes and reliability/recovery; pair with every prompt/context change.
- `[[distributed-systems-fundamentals]]` — failure models, idempotency, exactly-once, retries/backoff,
  timeouts: the foundation under durable execution / agent reliability (§7).
- `[[ml-observability-monitoring]]` — monitoring reliability signals (retry rates, parked/stuck workflows,
  resume/replay counts) alongside agent traces.
- `[[ai-security-on-gke]]` — guardrails, sandboxing untrusted tool/code execution, agent threat model.
- `[[serving-frameworks]]` — self-hosting models (vLLM/SGLang/Triton/KServe) + constrained decoding.
- `[[gke-inference-gateway]]` — model-aware routing/load-balancing/fallback in front of model replicas.
- `[[aiml-on-kubernetes]]` — running AI/ML (incl. agentic) workloads on K8s/GKE (umbrella).
- `[[kubernetes-expert]]` — general K8s deployment/scaling/secrets the agent service runs on.

---

# Reference — llm-app-agent-frameworks

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
- **Cache aggressively** (see §9 and §10): prompt/prefix caching on the provider side, plus your own
  response/embedding cache for deterministic sub-calls.
- **Watch the token bill compounding:** every agent turn re-sends growing history × every tool result ×
  every reflection round. Loops multiply tokens. Budget and cap (§8, §10).

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

## 7. Agent reliability & durable / long-running orchestration

A naive agent loop is an in-process `while` loop holding all of its state — the message history, the
plan, which tools already ran, partial results — in local variables. If the pod is rescheduled, the
process OOMs, an LLM call times out, or a tool 500s mid-run, **the entire run is lost** and there is no
safe way to resume: re-running from the top re-executes side effects (re-charges the card, re-sends the
email), re-running from a guessed point corrupts state. Agents make this worse than a normal service
because a single run is *long* (seconds to days), *stateful*, and *fans out to flaky dependencies*
(every LLM call and tool call can fail, time out, rate-limit, or return garbage). Reliability here is a
**distributed-systems problem**, not a prompt problem → [[distributed-systems-fundamentals]] (failure
models, idempotency, exactly-once, retries, timeouts). Bounding loops (§10) keeps a *single* run from
running away; durable orchestration keeps a run *alive and correct across failures*.

### 7.1 Durable execution — the answer to "what happens when it crashes"

**Durable execution** engines persist the progress of a workflow so it survives process death and
resumes exactly where it left off. The model: your orchestration code is a **workflow** whose every
step's input and result is **checkpointed to durable storage**; on crash/restart the engine **replays**
the workflow from its log, skipping already-completed steps (returning their persisted results) until it
reaches the point of failure, then continues. The agent's control flow becomes crash-proof without you
hand-rolling a state machine + database.

Engines you'll encounter (shapes differ; **verify current APIs/SDKs against each project's docs** — this
is fast-moving):

| Engine | Shape | Notes |
|---|---|---|
| **Temporal** | Workflow/activity SDK (Go/Java/Python/TS) + a server cluster; deterministic replay from an event history | The reference model for durable workflows; mature human-in-the-loop signals, timers, retries |
| **AWS Bedrock AgentCore** | Managed runtime for long-running/durable agents on AWS (sessions, memory, identity, tools) | Cloud-managed; verify the exact feature surface against current AWS docs |
| **Restate** | Durable execution + durable promises/state; lightweight single binary | Strong for durable RPC/handlers and idempotent service calls |
| **DBOS** | Durable workflows as decorated functions backed by Postgres | Library + Postgres, no separate cluster; "the database is the orchestrator" |
| **Inngest** | Event-driven durable functions/steps (steps are memoized/retried) | Step functions over events; good for async/background agent work |

Some agent frameworks add their own durability layer (e.g. LangGraph **checkpointers** persist graph
state for resume/human-in-the-loop). That covers *graph state*; a durable-execution engine covers
*arbitrary orchestration code + side effects*. They compose — run a LangGraph/ADK agent **inside** a
durable workflow when you need both.

### 7.2 The workflow vs. activity split (the core discipline)

Durable replay only works if the workflow is **deterministic**: replaying the same event history must
take the same path. Therefore:

- **Workflow code = deterministic orchestration only.** Control flow, sequencing, deciding which step
  runs next. **No I/O, no clocks, no randomness, no direct LLM/tool calls** in workflow code — those are
  non-deterministic and would diverge on replay.
- **Activities (a.k.a. steps) = all the non-deterministic, side-effecting work.** The LLM call, the
  tool/API call, the DB write, the retrieval. Each activity's result is checkpointed, so on replay it is
  *not re-executed* — the persisted result is returned.
- Get the engine's facilities for the things you'd otherwise do non-deterministically: **durable timers/
  sleeps** (not `time.sleep`), **engine-provided "now"/random**, **durable signals** for external input.

> The mental rule: **anything that can fail, block, or vary goes in an activity; the workflow just
> decides what to do with the results.** An agent loop maps cleanly: the loop/plan/edge-selection is
> workflow code; "call the model", "run tool X", "retrieve" are activities.

### 7.3 Idempotency, retries, and exactly-once side effects

Replay and retries mean **an activity may execute more than once**. This is the classic distributed-
systems trap (link [[distributed-systems-fundamentals]]): **at-least-once delivery + non-idempotent side
effect = double charge / duplicate email / double-provision.** Defenses:

- **Make every side-effecting tool idempotent.** Pass a stable **idempotency key** (derive it from the
  workflow/run id + step id, *not* a fresh UUID each attempt) so the downstream system dedupes a retry.
  Payment APIs, message sends, and provisioning calls almost all support idempotency keys — use them.
- **Retries with backoff belong on the activity, configured declaratively** (max attempts, initial
  interval, backoff coefficient, max interval, which errors are retryable vs. fatal). Don't hand-roll
  `for attempt in range(...)` inside an activity that the engine is also retrying — you'll double-retry.
- **Cap retries and total attempts** — unbounded retries against a persistently failing dependency burn
  cost and never converge. Distinguish *retryable* (timeout, 503, rate-limit) from *non-retryable* (4xx
  validation, auth) so you fail fast on the latter. This is the durable-orchestration analogue of the
  loop bounds in §10; both must exist.
- **Exactly-once is "effectively once":** the engine guarantees the *workflow* observes each activity's
  result once; the *world* is made consistent by idempotency + dedupe, not by magic. Design for it.
- **Timeouts and heartbeats:** set per-activity start-to-close and schedule-to-close timeouts so a hung
  LLM/tool call doesn't wedge the run forever. For long activities, **heartbeat** so the engine can tell
  "still working" from "dead" and retry only the dead ones.

### 7.4 Long-running & async agents

Durable execution is what makes genuinely long-lived agents safe:

- **Human-in-the-loop pause/resume.** The workflow blocks on a **durable signal/wait** (approve this
  refund, answer this question) — possibly for hours or days — consuming no compute while parked, and
  resumes deterministically when the signal arrives. This is the correct way to gate destructive actions
  (§6 / [[ai-security-on-gke]]) without holding a process open.
- **Durable timers for multi-day flows.** "Wait 24h, then follow up", scheduled re-checks, SLA timers —
  use the engine's durable sleep, not an in-process timer that dies with the pod.
- **Checkpoint/resume of agent memory & context.** Persist the working state (message history, plan,
  scratchpad, tool results, the context-engineering window from §9) at each step so a resumed run rebuilds
  the *exact* context — not an approximation. Keep large blobs out-of-band (store a handle/id), as in §9.
- **Recovery semantics, stated explicitly.** Decide and document: on resume do you *replay* (re-derive
  state from the event log — requires determinism) or *restore* (load a saved snapshot)? What's the unit
  of retry — a single tool call, or the whole step? What happens to in-flight side effects on crash
  (idempotency answers this)? An agent without an explicit recovery story will lose or corrupt work.

### 7.5 Running durable agents on Kubernetes → [[kubernetes-expert]]

The durability lives in the **backend** (the engine's persistent store / managed service), which frees
the compute tier to be ordinary, replaceable workers:

- **Workers are stateless and horizontally scalable.** Temporal/Restate/Inngest/DBOS workers poll the
  durable backend for work; any worker can pick up any task, and a killed worker's task is retried/
  resumed elsewhere. Run them as a **Deployment**, scale on backlog/queue depth or in-flight task count
  (KEDA), not CPU% → [[autoscaling-kubernetes]]. This is the same "externalize state" rule as §6 — the
  durable backend *is* the externalized state.
- **Prefer a durable backend over making the agent pod itself stateful.** A StatefulSet with local state
  reintroduces exactly the crash-loses-everything problem. Use StatefulSets/PVCs for the *datastore*
  (Postgres/etc.) or run the engine as a managed service; keep agent workers stateless.
- **Operational concerns:** size worker concurrency to the durable backend's throughput; isolate
  task queues per workload so a heavy agent doesn't starve others; pin SDK/server versions (replay
  determinism can be sensitive to SDK upgrades — verify each engine's versioning/patching guidance).
- Long autonomous runs still pair with the **async + queue + worker** shape from §6; the durable engine
  is the queue + state + retry machinery done correctly.

### 7.6 Observability & evaluating reliability

You cannot fix failure modes you can't see (general tracing in §10):

- **The durable history *is* a trace.** Each engine exposes the per-step event history (inputs, outputs,
  attempts, retries, timers, signals). Export it alongside your OTel GenAI span tree (§10) so one view
  shows both the agent trajectory and the durability events. Reliability signals → [[ml-observability-monitoring]].
- **Track reliability as metrics:** per-activity retry counts and failure rates, time-parked-on-signal,
  resume/replay counts, workflow completion vs. timeout vs. failure, cost per (possibly retried) run.
  Alert on retry storms and stuck/parked workflows.
- **Trajectory replay for eval.** Because the history is persisted, you can replay a failed run's
  trajectory to reproduce a bug deterministically, and build an eval set of real failure trajectories
  (bad tool selection, loops, recoveries) → [[ml-evaluation-evals]]. Evaluate *recovery* too: inject
  activity failures and assert the agent resumes correctly and side effects don't double-fire.

### 7.7 Anti-patterns (reliability)

- **In-memory agent loop with no persistence.** State lives only in process; a restart loses the run and
  there's no safe resume. The default failure of "we just ran the loop in our API handler."
- **Non-idempotent tool calls behind retries → double side effects.** Retrying or replaying a charge/
  send/provision with no idempotency key → duplicates. The single most damaging durable-agent bug
  ([[distributed-systems-fundamentals]]).
- **Unbounded retries / cost.** Retrying a permanently-failing dependency forever; not separating
  retryable from fatal errors; no cap on attempts or total run cost.
- **No resume / no recovery story.** Crash → re-run from scratch (re-doing side effects) or manual
  surgery. Decide replay-vs-restore and unit-of-retry *before* shipping.
- **Hidden non-determinism breaking replay.** Calling the LLM/tool, reading the clock, using `random`, or
  iterating a hash-ordered map *in workflow code* → replay diverges and the engine errors or corrupts
  state. Keep all of that in activities; use engine-provided time/random/signals.
- **Stateful agent pods as the durability mechanism.** Pinning runs to a pod's local memory/disk instead
  of a durable backend — you've just rebuilt the in-memory-loop problem with extra steps.

---

## 8. RAG integration (brief — depth in [[rag-vector-databases]])

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

## 9. Prompt & context engineering

Prompt and context engineering are the two highest-leverage disciplines in an LLM app — more so than
framework choice. **Prompt engineering** is how you write the instructions; **context engineering** is
how you decide *what goes into the window at all*, and in what order, on every turn. The model only
ever sees the tokens you assemble. Treat that assembly as a first-class, versioned, evaluated system —
not string concatenation buried in code.

### 9.1 Prompt engineering

- **System vs. user prompt.** The **system** prompt carries durable role, policy, format rules, and
  tool-use guidance — it's stable and should be the cacheable prefix (§10). The **user** turn carries the
  variable task. Don't smuggle per-request data into the system prompt (it breaks prefix caching and
  blurs the trust boundary). Keep developer/system instructions and untrusted user/tool content clearly
  separated — never concatenate retrieved or tool-returned text into the instruction region.
- **Few-shot / in-context learning.** Provide 1–N worked examples to pin format and behavior. Few-shot
  is most valuable for *format/style* and edge-case handling; for pure capability, a clearer instruction
  often beats more examples. Keep examples diverse, correct, and representative of hard cases; order
  matters (recency bias — the last example carries weight). Strip few-shot down as the instruction
  matures — examples are expensive tokens on every call. For dynamic few-shot, retrieve the *k* most
  similar examples per query (this overlaps with RAG → [[rag-vector-databases]]).
- **Chain-of-thought & structured reasoning.** Asking the model to reason step-by-step before answering
  improves multi-step/math/logic tasks. Modern **reasoning models** do this natively — for those, do
  *not* hand-roll "think step by step" scaffolding or force a visible scratchpad; give them the goal and
  constraints and let them reason. For non-reasoning models, explicit CoT, decomposition, or
  self-consistency (sample several reasoning paths, take the majority) still help. When you need both
  reasoning *and* clean structured output, separate the reasoning channel from the final answer (e.g. a
  `reasoning` field or a thinking phase) so parsing isn't polluted. **Reasoning is fast-moving — verify
  current model behavior and the provider's reasoning/thinking API against current docs.**
- **Role / format / delimiter discipline.** State the role, the task, the constraints, and the exact
  output format explicitly. Delimit sections with stable, unambiguous markers (XML-ish tags, headings,
  fenced blocks) so the model — and your parser — can tell instructions from data from examples. Put the
  most important instruction where it's least likely to be lost (start and end; see "lost in the middle"
  below). Be positive and specific ("respond in ≤3 sentences") over vague negatives.
- **Output schemas / structured output.** When you need machine-parseable output, **constrain, don't
  hope** — native JSON-schema / grammar / tool-as-schema, then validate (full treatment in §2 and
  `examples.md`). Describe each field in the schema; the schema *is* prompt.
- **Prompt templating & versioning.** Prompts are code: parameterized templates, under source control,
  reviewed, and tied to eval results (§10 evaluation, [[ml-evaluation-evals]]). Externalize them so you
  can iterate/roll back without redeploying, but pin which prompt version a deployment runs. Never
  string-concatenate logic into prompts ("prompt spaghetti", §11).
- **Decoding params.** `temperature` and `top_p` (nucleus) control randomness — lower (≈0) for
  extraction/classification/tool-arg formatting and anything you'll parse; higher for creative
  generation. Tune *one* of temperature or top_p, not both. `max_tokens` bounds cost and runaway output;
  `stop` sequences cut generations cleanly; penalties (frequency/presence) curb repetition. Pick params
  per node, not globally. **Reasoning models often ignore or restrict these knobs — verify per model.**
- **Determinism & caching.** Even at temperature 0, outputs are *not* bit-exact (batching/kernel
  effects) — never depend on exact-string reproducibility. For sub-tasks that should be stable, cache by
  a hash of the exact input. Order the prompt so the long *stable* prefix (system prompt, tool defs,
  pinned context) comes first to maximize provider/engine **prefix caching** (§10). Make cache hit rate a
  tracked cost metric.
- **Prompt injection is a security property, not a prompt trick.** Any instruction-looking text inside
  untrusted content (tool output, retrieved docs, MCP results, user-supplied files, web pages) can
  hijack the model ("ignore previous instructions", exfiltration, unintended tool calls). You **cannot**
  fully solve this with prompt wording. Mitigate in layers: keep untrusted content out of the
  instruction region and clearly delimited as data; least-privilege tools; human-gate destructive
  actions; output/egress guardrails; sandbox tool execution. Full threat model → [[ai-security-on-gke]].

### 9.2 Context engineering — the context window is a managed resource

The bigger modern idea: the context window is **finite, attention is non-uniform across it, and every
token costs money and can dilute the rest.** Context engineering is the discipline of curating *exactly*
the right tokens for each model call. More context is not better — beyond a point it *hurts* (cost,
latency, and quality). **This is a fast-moving best-practice area — verify against current writing from
model providers and tooling before treating any specific tactic or limit as settled.**

- **Retrieval & selection — pick what goes in.** Don't dump everything you have; select the minimal
  sufficient context for *this* turn: the relevant docs (RAG → [[rag-vector-databases]]), the relevant
  memories, the relevant tool results, the relevant slice of history. Relevance-rank and cap each
  source. Selection quality is usually a bigger lever than model size.
- **Ordering & "lost in the middle."** Models attend most strongly to the **beginning and end** of a
  long context and can miss material buried in the middle. Put the task/question and the highest-value
  context at the edges; don't rely on a critical fact sitting at position 4,000 of 12,000. Re-ranking so
  the best retrieved chunks sit at the top/bottom matters as much as retrieving them.
- **Compression & summarization.** When history or tool output grows, compress it: summarize old turns
  into a running synopsis, extract just the facts you need from a large tool result, drop superseded
  content. Protect the system prompt and active task framing from eviction. Summarization is lossy —
  summarize *deliberately* (keep ids/citations/decisions), and keep the raw artifact retrievable if the
  agent might need it again.
- **Memory & state across turns.** Short-term (working) memory = what's in the window now; manage it via
  trimming/summarization. Long-term memory = facts/preferences/episodes persisted *outside* the window
  and retrieved on demand (semantic / episodic / procedural — see §2 Memory). Write to memory
  deliberately (not everything), retrieve selectively, and **treat retrieved memory as untrusted input.**
- **Tool-result management.** Tool/observation output is the fastest way to blow the budget. Return
  terse, structured observations; truncate or summarize large payloads before they enter context; keep
  large blobs out-of-band (store them, pass a handle/id the agent can re-fetch). Prune stale
  observations from earlier steps once they've served their purpose.
- **Long-context vs. RAG.** Large context windows don't make retrieval obsolete. Stuffing a huge corpus
  into the window costs more, adds latency, and degrades quality (lost-in-the-middle, dilution) versus
  retrieving the relevant slice. Rule of thumb: **retrieve to narrow, use long context for the
  narrowed-down material** and for genuinely whole-document reasoning. Measure both with evals
  ([[ml-evaluation-evals]]) rather than assuming. RAG depth → [[rag-vector-databases]].
- **Token / cost budgeting.** Set an explicit context budget per call and per run (system + tools +
  retrieved + history + headroom for the response). Every agent turn re-sends growing history × tool
  results × reflection rounds — loops multiply tokens (§5, §10). Track tokens per node and per session as
  a first-class metric; right-size the model per node so cheap nodes don't pay frontier prices.
- **Context rot.** Over a long session, context accumulates stale, contradictory, or low-value tokens
  that quietly degrade quality and inflate cost ("context rot"). Counter it: periodically compact/
  re-summarize, drop superseded content, re-anchor the task framing, or start a fresh window seeded with
  a clean summary + the live state. Don't let a session's window grow monotonically forever.

### 9.3 How this couples to evals and RAG

- **Evals are how you know any of this works.** Prompt edits, few-shot sets, decoding params, context-
  selection strategy, compression thresholds, and long-context-vs-RAG choices are all changes you must
  measure, not guess. Hold a versioned offline set; A/B prompt and context strategies; track quality vs.
  tokens/cost/latency. Every prompt or context change re-runs evals in CI (§10). Full eval discipline →
  [[ml-evaluation-evals]].
- **RAG is context engineering applied to a corpus.** Chunking, embeddings, hybrid retrieval, and
  re-ranking are all in service of *putting the right tokens in the window* — the selection/ordering/
  compression principles above are exactly the RAG quality levers. Defer the retrieval mechanics to
  [[rag-vector-databases]]; treat every retrieved span as untrusted input (§9.1, §8).

---

## 10. Production concerns

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

## 11. Anti-patterns (these cause the production incidents)

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
- **In-memory loop with no durability (for long/stateful agents).** State only in process → a crash
  loses the run with no safe resume; non-idempotent tool calls + retries/replay → double side effects.
  Use durable execution for long-running agents (§7) → [[distributed-systems-fundamentals]].

---

## Rationalizations & rebuttals

The excuses an engineer or agent uses to skip the right thing when building an LLM app — each rebutted.

- **"No eval needed — it demos fine."** A demo is one trajectory; production is a distribution. Without a
  versioned eval set you regress silently on every prompt/model bump and learn from users (§10). Build
  the dataset *before* the agent.
- **"The unbounded agent loop is fine, it always terminates in testing."** "Always" until an injection,
  an oscillation, or a flaky tool makes it spin — one stuck session can burn thousands of dollars (§10,
  §11). Every loop gets a hard cap: max steps, tokens, wall-clock, and cost, plus degenerate-loop
  detection.
- **"Tool output is from our own systems, so I can trust it."** Tool, RAG, and MCP results are untrusted
  input — any instruction-looking text in them can hijack the model ("ignore previous instructions",
  exfiltration) (§4, §8, §9.1). Keep it out of the instruction region, delimited as data; you cannot fix
  this with prompt wording.
- **"In-memory state is fine, the run is short."** Short until a pod reschedule, OOM, or tool timeout
  loses the whole run with no safe resume — and re-running re-fires side effects (§7). For long/stateful
  agents use a durable backend; decide replay-vs-restore *before* shipping.
- **"No sandbox — the model-generated code/tool call looks harmless."** One injection and the agent acts
  with your credentials. Never run generated code or net/fs tools in-process or with cluster creds;
  isolate (gVisor/microVM), least-privilege, egress-restrict, human-gate destructive actions (§6, §10).
- **"Retries are safe, the engine guarantees exactly-once."** The engine guarantees the *workflow*
  observes a result once; the *world* is made consistent by idempotency, not magic (§7.3). At-least-once
  + a non-idempotent charge/send = duplicates. Pass a stable idempotency key derived from run+step id.
- **"A multi-agent swarm will be more capable."** More agents = more tokens, more failure modes, harder
  evals and debugging (§2). Use the least autonomy that works; don't reach for multi-agent until a single
  well-tooled agent demonstrably can't do the job.
- **"Bigger context window means I can skip retrieval and just stuff everything in."** More context costs
  more, adds latency, and degrades quality (lost-in-the-middle, dilution, context rot) (§9.2). Retrieve
  to narrow, then use long context for the narrowed material — and measure both with evals.

---

## Red flags

Stop and reconsider if you see any of these:

- **No loop bounds.** A loop without a max-steps / max-tokens / wall-clock / max-cost cap or
  degenerate-loop detection — runaway cost and hangs waiting to happen (§10, §11).
- **No evals.** Prompt/model/framework changes shipped on vibes, no versioned offline set, nothing
  running in CI. "No evals = no production" (§10).
- **Untrusted exec with no sandbox.** Model-generated code or network/filesystem tools running in-process
  or with cluster/cloud credentials; destructive tools with no human gate (§6, §10).
- **Non-idempotent side-effecting tools behind retries/replay.** Charges, sends, or provisioning calls
  with no idempotency key — the single most damaging durable-agent bug (§7.3).
- **In-memory loop holding all run state.** State only in local variables for a long/stateful run; no
  durable backend, no defined recovery story (replay vs. restore, unit-of-retry) (§7).
- **Prompt spaghetti.** Giant, unversioned, string-concatenated prompts with business logic buried in
  them; untrusted tool/retrieved text concatenated into the instruction region (§9.1, §11).
- **No tracing / observability.** Debugging an agent loop with no per-step span tree (LLM/tool/retrieval
  inputs, outputs, tokens, latency, cost) — you can't reason about a loop you can't see (§10).
- **Giant or overlapping tool sets.** 60 tools where 5–15 orthogonal ones would do; selection accuracy
  degrades — route or split per sub-agent (§2).
- **Hidden non-determinism.** Depending on exact output strings, unpinned model ids, or non-deterministic
  calls (LLM/tool/clock/`random`) inside durable workflow code (§7.2, §10).

---

## Verification gate (definition of done)

Before an LLM app/agent counts as shippable, confirm:

- [ ] **Loops and cost are bounded.** Every loop has a hard cap (max steps, tool calls, tokens,
  wall-clock, per-run cost budget); a tripped cap fails gracefully (best-effort + flag), and degenerate
  loops (repeated tool+args, oscillation) are detected and aborted (§10).
- [ ] **Evals exist and gate changes.** A versioned offline eval set covers final-answer, tool-selection,
  retrieval, and trajectory; it runs in CI on every prompt/model/framework change; online signals
  (success/abandonment, guardrail hits, cost/latency, loop counts) are tracked (§10).
- [ ] **Untrusted execution is sandboxed.** Model-generated code and net/fs tools run isolated
  (gVisor/microVM), least-privilege, egress-restricted, no node/cluster creds; destructive actions are
  human-gated; all tool/RAG/MCP output is handled as untrusted, delimited data (§6, §9.1, §10).
- [ ] **Side-effecting tools are idempotent.** Args validated server-side; a stable idempotency key
  (derived from run+step id) is passed to every charge/send/provision call; retries are bounded and split
  retryable vs. fatal (§2, §7.3).
- [ ] **State is durable/resumable where the run is long or stateful.** Externalized session/state or a
  durable-execution backend; an explicit recovery story (replay vs. restore, unit-of-retry); workflow
  code kept deterministic so replay holds; stateless, horizontally scalable workers (§6, §7).
- [ ] **Tracing is on.** Every run is a span tree (LLM/tool/retrieval/sub-agent spans with inputs,
  outputs, token counts, latency, cost), ideally via OpenTelemetry GenAI conventions; prompts are
  versioned with each trace so regressions reproduce; secrets are redacted from logs/traces (§10).
- [ ] **Prompts and models are pinned and versioned.** Prompts under source control, tied to eval
  results; model ids pinned; evals re-run on any bump (§9.1, §10).

---

## 12. Version awareness

This space changes monthly. Model names, context limits, pricing, structured-output support, MCP spec
revisions and transport details, OpenTelemetry GenAI convention attribute names, and every framework's
API (especially LangChain/LangGraph, ADK, LlamaIndex, CrewAI, AutoGen) all move. **Before relying on
any specific class name, method signature, config key, model id, or limit in this guide, check the
current official docs.** The *patterns* here are durable; the *surfaces* are not.

---

## 13. Canonical references (verify currency)

- **Model Context Protocol** — spec & docs: https://modelcontextprotocol.io and
  https://spec.modelcontextprotocol.io
- **Google ADK** — https://google.github.io/adk-docs/
- **LangGraph / LangChain** — https://langchain-ai.github.io/langgraph/ and https://python.langchain.com/
- **LlamaIndex** — https://docs.llamaindex.ai/
- **CrewAI** — https://docs.crewai.com/ · **AutoGen** — https://microsoft.github.io/autogen/ ·
  **Pydantic-AI** — https://ai.pydantic.dev/ · **Haystack** — https://docs.haystack.deepset.ai/
- **OpenTelemetry GenAI semantic conventions** —
  https://opentelemetry.io/docs/specs/semconv/gen-ai/
- **Durable execution engines** (verify current SDK/API): Temporal https://docs.temporal.io/ ·
  AWS Bedrock AgentCore https://docs.aws.amazon.com/bedrock-agentcore/ · Restate https://docs.restate.dev/ ·
  DBOS https://docs.dbos.dev/ · Inngest https://www.inngest.com/docs ·
  LangGraph persistence/checkpointers https://langchain-ai.github.io/langgraph/concepts/persistence/
- **gVisor (runsc sandbox)** — https://gvisor.dev/docs/
- Anthropic, "Building effective agents" — https://www.anthropic.com/research/building-effective-agents
- ReAct paper — https://arxiv.org/abs/2210.03629 · Reflexion — https://arxiv.org/abs/2303.11366
- **Prompt & context engineering** — provider prompting guides (verify current): Anthropic
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview · OpenAI
  https://platform.openai.com/docs/guides/prompt-engineering · Google
  https://ai.google.dev/gemini-api/docs/prompting-strategies
- "Lost in the Middle: How Language Models Use Long Contexts" — https://arxiv.org/abs/2307.03172
- Chain-of-Thought prompting — https://arxiv.org/abs/2201.11903 · Self-Consistency —
  https://arxiv.org/abs/2203.11171

---

# Agent Patterns — Worked Examples

Canonical, correct-in-shape artifacts to imitate. **The frameworks/SDKs below move fast — treat exact
class names, method signatures, and config keys as illustrative of the shape, and verify against the
current official docs before relying on them.** Imports are partially elided for brevity.

---

## 1. A LangGraph tool-calling agent with typed state + bounded loop

A minimal ReAct-style agent built on the control-flow-as-graph model: typed shared state, an LLM node,
a tool node, a conditional edge that loops until the model stops calling tools — with a **hard step
cap** so it can never run away. Pattern shown for LangGraph; the same shape applies in ADK (agent +
tools + runner + session).

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
# Model via an OpenAI-compatible endpoint — swap base_url/key for hosted vs self-hosted (vLLM/SGLang).
from langchain_openai import ChatOpenAI

MAX_STEPS = 8  # hard loop bound — non-negotiable in production

# --- Tools: validate args server-side; the model's arguments are UNTRUSTED. ---
@tool
def get_weather(city: str) -> str:
    """Return current weather for a city. Use only for present-day weather questions."""
    city = city.strip()
    if not city or len(city) > 100:        # validate; never trust the model's JSON blindly
        raise ValueError("city must be a non-empty string under 100 chars")
    # ... call a real weather API here; return a TERSE, structured observation ...
    return f'{{"city": "{city}", "temp_c": 21, "condition": "clear"}}'

TOOLS = [get_weather]

# --- Typed shared state flows through every node. `steps` enforces the loop bound. ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int

# Bind tools so the model emits native structured tool calls (not text-scraped ReAct).
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
SYSTEM = SystemMessage(content="You are a concise assistant. Use tools when needed; otherwise answer.")

def call_model(state: AgentState) -> dict:
    resp = llm.invoke([SYSTEM, *state["messages"]])
    return {"messages": [resp], "steps": state["steps"] + 1}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if state["steps"] >= MAX_STEPS:        # bound tripped -> stop, return best effort
        return END
    return "tools" if getattr(last, "tool_calls", None) else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))   # executes tool calls, appends observations to state
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")           # loop: observation -> model -> ...

# Checkpointer externalizes state so any replica can resume a thread (stateless service, see §6).
# from langgraph.checkpoint.postgres import PostgresSaver
# app = graph.compile(checkpointer=PostgresSaver(...))
app = graph.compile()

result = app.invoke(
    {"messages": [("user", "What's the weather in Paris?")], "steps": 0},
    config={"recursion_limit": 25},        # belt-and-suspenders: framework-level run cap too
)
print(result["messages"][-1].content)
```

Why this is the GOOD shape: explicit graph + typed state; native tool calling; tool **validates its own
input**; observations are terse; the loop is bounded *twice* (app-level `steps` cap and framework
`recursion_limit`); state is externalizable via a checkpointer for a stateless, horizontally-scaled
deployment.

---

## 2. Structured output: constrain, then validate

Don't hope the model returns JSON — define the schema and validate the result.

```python
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    category: str = Field(description="one of: bug, billing, feature, other")
    priority: int = Field(ge=1, le=5)
    summary: str

# Native structured output: the provider constrains generation to the schema.
structured = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Ticket)
ticket = structured.invoke("Customer says checkout 500s on every card. Triage it.")
assert isinstance(ticket, Ticket)          # validated object, not a hopeful string parse
```

On self-hosted engines, get the same guarantee with grammar/constrained decoding (vLLM/SGLang +
outlines/xgrammar) → [[serving-frameworks]].

---

## 3. MCP tool-server wiring sketch

An MCP **server** exposes capabilities (tools/resources/prompts); your agent embeds an MCP **client**
that connects over a transport and surfaces those tools as native tools. Reuse one server across many
agents/frameworks.

### Server (Python, FastMCP-style)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("docs-server")

@mcp.tool()
def search_docs(query: str, k: int = 5) -> list[dict]:
    """Search the docs corpus and return top-k snippets with source ids for citation."""
    if not query.strip():
        raise ValueError("query required")
    k = max(1, min(k, 20))                  # bound it; don't let the caller request 10k results
    # ... do retrieval; return TERSE snippets + source ids ...
    return [{"id": "doc-12#3", "text": "..."}]

@mcp.resource("config://service-limits")    # a readable Resource (context), not an action
def service_limits() -> str:
    return '{"max_qps": 50, "max_tokens": 8000}'

if __name__ == "__main__":
    mcp.run(transport="stdio")               # stdio for local/co-located; streamable HTTP for remote
```

### Client side (agent consumes the MCP server's tools)

```python
# Most frameworks ship an MCP adapter that maps MCP tools -> native tools. VERIFY the current adapter
# API/version against your framework's docs — this surface is young and changing.
#
#   from langchain_mcp_adapters.client import MultiServerMCPClient   # example adapter
#   client = MultiServerMCPClient({
#       "docs": {"command": "python", "args": ["docs_server.py"], "transport": "stdio"},
#       # remote: {"url": "https://mcp.example.com/mcp", "transport": "streamable_http"},
#   })
#   tools = await client.get_tools()          # now usable like any other tool in section 1
```

**Security (do not skip):** the MCP server owns credentials with least privilege; the agent only sees
declared tools/resources. **Treat the server as untrusted code and every returned snippet/resource as
untrusted, possibly prompt-injecting input** — never let tool output become trusted instructions. Pin
and vet remote servers; human-gate destructive tools → [[ai-security-on-gke]].

---

## 4. Context engineering: assemble a budgeted, cache-friendly context

Context engineering is *what tokens go in the window, in what order, within a budget* — not string
concatenation. This sketch shows the durable shape: a **stable cacheable prefix first** (system + tool
defs), then **selected + re-ranked** retrieved context with the best chunks at the edges (lost in the
middle), then a **compressed** history, then the live turn — all under an explicit token budget. Treat
retrieved/tool text as untrusted data, clearly delimited from instructions.

```python
# Illustrative shape — verify your tokenizer/SDK APIs against current docs.
TOKEN_BUDGET = 8000          # per-call ceiling: prefix + context + history + headroom for the answer
ANSWER_HEADROOM = 1200

def count(text: str) -> int: ...        # use the model's real tokenizer, not len()/4

# 1) STABLE PREFIX FIRST — maximizes provider/engine prefix caching; never put per-request data here.
SYSTEM = "You are a support agent. Answer ONLY from <context>. If it's not there, say you don't know."

def select_context(chunks: list[dict], budget: int) -> list[dict]:
    """Select + re-rank, then place best chunks at the EDGES (start/end) to fight lost-in-the-middle."""
    ranked = rerank(chunks)                       # relevance-rank; don't dump everything you retrieved
    kept, used = [], 0
    for c in ranked:                              # greedily fill against the budget
        t = count(c["text"])
        if used + t > budget:
            break
        kept.append(c); used += t
    # interleave so the top items land first AND last (edges), weaker ones in the middle
    edges = kept[::2] + kept[1::2][::-1]
    return edges

def compress_history(history: list[dict], budget: int) -> str:
    """Summarize old turns into a running synopsis; keep ids/decisions; protect the task framing."""
    if count(render(history)) <= budget:
        return render(history)
    return summarize(history, keep=["decisions", "ids", "open_questions"])   # lossy ON PURPOSE

def build_messages(query: str, retrieved: list[dict], history: list[dict]) -> list[dict]:
    ctx_budget  = TOKEN_BUDGET - count(SYSTEM) - count(query) - ANSWER_HEADROOM
    hist_budget = ctx_budget // 3                 # explicit split: history gets a slice, context the rest
    ctx_chunks  = select_context(retrieved, ctx_budget - hist_budget)

    # Untrusted retrieved text is DATA, fenced and tagged — never merged into the instruction region.
    context_block = "\n".join(f'<doc id="{c["id"]}">{c["text"]}</doc>' for c in ctx_chunks)
    return [
        {"role": "system", "content": SYSTEM},                       # stable, cacheable
        {"role": "system", "content": f"<context>\n{context_block}\n</context>"},
        {"role": "system", "content": f"<history>\n{compress_history(history, hist_budget)}\n</history>"},
        {"role": "user",   "content": query},                        # variable task last
    ]
```

Why this is the GOOD shape: a stable prefix maximizes prefix-cache hits (cost/latency); retrieval is
**selected and budgeted**, not dumped; the best chunks sit at the **edges** (lost-in-the-middle);
history is **compressed** so the window doesn't rot; untrusted retrieved content is fenced as data with
source ids for citation/grounding. Measure selection/ordering/compression choices with evals
([[ml-evaluation-evals]]); retrieval mechanics (chunking/embeddings/hybrid/re-rank) →
[[rag-vector-databases]]; injection threat model → [[ai-security-on-gke]].

---

## 5. Sandboxed tool execution on GKE (untrusted code/tools)

When a tool runs model-generated code or touches the network/filesystem, isolate it. Run the sandbox in
its own pod under the **gVisor (`runsc`) runtime class**, with no credentials, restricted egress,
non-root, read-only root FS, and tight limits. Never execute model-generated code in the agent process
or with cluster permissions.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tool-sandbox
  labels: { app: agent-tool-sandbox }
spec:
  runtimeClassName: gvisor          # gVisor (runsc) — kernel-level isolation for untrusted code
  automountServiceAccountToken: false   # no K8s API creds in the sandbox
  containers:
    - name: executor
      image: registry.example.com/tool-executor:pinned-digest
      command: ["/sandbox/run"]     # executes the model-supplied snippet, returns terse result
      securityContext:
        runAsNonRoot: true
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities: { drop: ["ALL"] }
        seccompProfile: { type: RuntimeDefault }
      resources:
        requests: { cpu: "250m", memory: "256Mi" }
        limits:   { cpu: "1",    memory: "512Mi" }   # cap blast radius + cost of a runaway tool
      volumeMounts:
        - { name: scratch, mountPath: /tmp }
  volumes:
    - name: scratch
      emptyDir: { sizeLimit: "64Mi" }    # ephemeral; nothing persists between runs
  # Pair with a NetworkPolicy that default-denies egress (allow only the specific hosts the tool needs).
```

The agent service is a separate, stateless Deployment that dispatches code to short-lived sandbox pods
(or a sandbox service), enforces a wall-clock timeout, and treats the returned output as untrusted.
Full agent threat model, guardrails, and egress controls → [[ai-security-on-gke]]; K8s deployment/
scaling → [[aiml-on-kubernetes]] / [[kubernetes-expert]].

---

## 6. Durable agent: workflow vs. activity split + idempotent tools

A crash-proof agent loop. The **workflow** is deterministic orchestration only — it decides what runs
next; it never calls the model/tools, reads the clock, or uses randomness directly. Every
non-deterministic, side-effecting step is an **activity** whose result is checkpointed, so on
replay/restart completed activities are *not* re-run. Shown in the shape of Temporal's Python SDK;
Restate / DBOS / Inngest differ in API but enforce the same split — **verify the current SDK against
the engine's docs.** See §7 of the guide; distributed-systems foundations → [[distributed-systems-fundamentals]].

```python
# Illustrative shape — verify the engine's current SDK/decorators against its docs.
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

# --- ACTIVITIES: all non-determinism + side effects live here (checkpointed, skipped on replay). ---
@activity.defn
async def call_model(messages: list[dict]) -> dict:
    # The LLM call is flaky + non-deterministic → must be an activity, never in workflow code.
    return await llm_client.chat(messages)          # returns {"tool_calls": [...]} or {"final": "..."}

@activity.defn
async def issue_refund(order_id: str, idempotency_key: str) -> dict:
    # SIDE EFFECT. Idempotency key derived from workflow+step id (passed in), NOT a fresh uuid here,
    # so a retry/replay dedupes downstream instead of double-refunding → exactly-once in effect.
    return await payments.refund(order_id, idempotency_key=idempotency_key)

# --- WORKFLOW: deterministic orchestration ONLY. No I/O, no clock, no random, no direct LLM/tool calls. ---
@workflow.defn
class RefundAgent:
    def __init__(self) -> None:
        self._approved: bool | None = None

    @workflow.signal
    def approve(self, ok: bool) -> None:            # durable signal: human-in-the-loop input
        self._approved = ok

    @workflow.run
    async def run(self, order_id: str) -> str:
        messages = [{"role": "user", "content": f"Process refund for {order_id}"}]
        retry = RetryPolicy(maximum_attempts=4, initial_interval=timedelta(seconds=1),
                            backoff_coefficient=2.0)   # bounded retries+backoff ON THE ACTIVITY

        for step in range(8):                        # hard loop bound (guide §10) — still required
            resp = await workflow.execute_activity(
                call_model, messages,
                start_to_close_timeout=timedelta(seconds=60),   # hung LLM call can't wedge the run
                retry_policy=retry,
            )
            if resp.get("final"):
                return resp["final"]

            # Gate the destructive action on a durable signal — parks for as long as needed, no compute.
            await workflow.wait_condition(lambda: self._approved is not None)
            if not self._approved:
                return "refund rejected by human"

            # Idempotency key from the deterministic workflow context — stable across retries/replay.
            key = f"{workflow.info().workflow_id}:refund:{step}"
            result = await workflow.execute_activity(
                issue_refund, args=[order_id, key],
                start_to_close_timeout=timedelta(seconds=30), retry_policy=retry,
            )
            messages.append({"role": "tool", "content": str(result)})
        return "step budget exhausted"               # fail gracefully, don't spin
```

Why this is the GOOD shape: the workflow survives worker/pod crashes and resumes by replaying the
event history (completed activities return their checkpointed results, not re-executed); the LLM and
refund calls are **activities** with bounded retries/backoff and timeouts; the refund is **idempotent**
via a key derived from the deterministic workflow context, so a retry or replay never double-refunds
([[distributed-systems-fundamentals]]); the human approval is a **durable signal** that parks the run
with zero compute; the loop is still **bounded** (§10). Workers are stateless and scale against the
durable backend on K8s → [[kubernetes-expert]] / [[autoscaling-kubernetes]]; export the durable history
alongside OTel GenAI spans → [[ml-observability-monitoring]].
