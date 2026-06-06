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
