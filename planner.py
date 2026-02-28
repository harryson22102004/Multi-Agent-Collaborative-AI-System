from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import AgentState

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior technical planner. Break down the user's task
into clear, ordered sub-tasks. Output a numbered list of steps.
Each step should be actionable and specific."""),
    ("human", "Task: {task}"),
])

class PlannerAgent:
    def __init__(self, model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.chain = PLANNER_PROMPT | self.llm

    def __call__(self, state: AgentState) -> AgentState:
        response = self.chain.invoke({"task": state["task"]})
        steps = [line.strip() for line in response.content.split("\n")
                 if line.strip() and line.strip()[0].isdigit()]
        return {
            **state,
            "plan": steps,
            "status": "researching",
            "messages": state["messages"] + [response],
        }
