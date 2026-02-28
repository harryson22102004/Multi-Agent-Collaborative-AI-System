from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

REVIEWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior code reviewer. Review the code for:
1. Correctness and edge cases
2. Security vulnerabilities
3. Performance issues
4. Code quality and best practices
Respond with either "APPROVED" or specific feedback for improvements."""),
    ("human", "Code to review:\n{code}"),
])

class ReviewerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.chain = REVIEWER_PROMPT | self.llm

    def __call__(self, state: AgentState) -> AgentState:
        response = self.chain.invoke({"code": state["code"]})
        approved = "APPROVED" in response.content.upper()
        return {
            **state,
            "review_feedback": response.content,
            "status": "done" if approved else "coding",
            "iteration": state["iteration"] + 1,
            "messages": state["messages"] + [response],
        }
