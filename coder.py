from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CODER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert software engineer. Write production-quality
code based on the plan and research. Include:
- Type hints, docstrings, error handling
- Unit tests
- Clear comments
If there's review feedback, address all issues."""),
    ("human", "Plan:\n{plan}\nResearch:\n{research}\nFeedback:\n{feedback}"),
])

class CoderAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.chain = CODER_PROMPT | self.llm

    def __call__(self, state: AgentState) -> AgentState:
        response = self.chain.invoke({
            "plan": "\n".join(state["plan"]),
            "research": state["research"],
            "feedback": state.get("review_feedback", "None"),
        })
        return {
            **state,
            "code": response.content,
            "status": "reviewing",
            "messages": state["messages"] + [response],
        }
