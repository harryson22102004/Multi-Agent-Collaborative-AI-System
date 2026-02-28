from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.tools.web_search import tavily_search

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research specialist. Given a plan, gather
relevant technical information, best practices, and code examples.
Synthesize findings into a structured report."""),
    ("human", "Plan:\n{plan}\n\nPrevious research: {research}"),
])

class ResearcherAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.chain = RESEARCH_PROMPT | self.llm

    def __call__(self, state: AgentState) -> AgentState:
        plan_text = "\n".join(state["plan"])
        # Search for relevant info
        search_results = tavily_search(state["task"])
        response = self.chain.invoke({
            "plan": plan_text,
            "research": search_results,
        })
        return {
            **state,
            "research": response.content,
            "status": "coding",
            "messages": state["messages"] + [response],
        }
