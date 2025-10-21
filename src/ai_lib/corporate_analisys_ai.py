from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.common.prompt import CORPORTATE_ANALYSIS_AI_SYSTEM_PROMPT
from src.tools.fetch import fetch_ddg_page
from src.tools.search import search_ddg


class CorporateAnalysisAI:

    def __init__(self, llm):
        self.agent = self.create_agent(llm)

    @classmethod
    def create_agent(cls, llm):
        tools = [search_ddg, fetch_ddg_page]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CORPORTATE_ANALYSIS_AI_SYSTEM_PROMPT),
                # MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=30,
        )

    def __call__(self, prompt: str) -> str:
        response = self.agent.invoke({"input": prompt})
        return response["output"]