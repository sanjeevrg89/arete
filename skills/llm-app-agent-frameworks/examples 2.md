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

## 4. Sandboxed tool execution on GKE (untrusted code/tools)

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
