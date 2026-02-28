from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.coder import CoderAgent
from src.agents.reviewer import ReviewerAgent

def should_continue(state: AgentState) -> str:
    if state["status"] == "done":
        return "done"
    if state["iteration"] >= 3:
        return "done"  # Max retries
    return state["status"]

def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("planner", PlannerAgent())
    graph.add_node("researcher", ResearcherAgent())
    graph.add_node("coder", CoderAgent())
    graph.add_node("reviewer", ReviewerAgent())

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_conditional_edges("reviewer", should_continue, {
        "coding": "coder",
        "done": END,
    })
    return graph.compile()
