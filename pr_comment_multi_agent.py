from os import getenv

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.ai_lib.corporate_analisys_ai import CorporateAnalysisAI
from src.ai_lib.pr_ai import PrAi
from src.common.prompt import PR_AI_MULTI_AGENT_SYSTEM_PROMPT
from src.tools.fetch import fetch_ddg_page
from src.tools.search import search_ddg

load_dotenv(verbose=True)
corporate_analysis_agent = CorporateAnalysisAI.create_agent(
    ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
)
pr_ai_agent = PrAi.create_agent(
    ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
)


@tool(
    description="このツールは、企業に関する情報を収集・分析し、自己PR作成のための洞察を提供します。入力として企業名だけを受け取り、関連するデータを収集・分析して、自己PRに役立つ情報を出力します。",
)
def call_corporate_analysis_agent(query: str):
    response = corporate_analysis_agent.invoke(
        {"input": query}
    )
    return response["output"]


@tool(
    description="このツールは、企業の情報もとに、効果的な自己PRを作成するためのサポートを提供します。入力として企業の情報を受け取り、受け取ったデータをもとに自己PRを生成します。通常、企業調査ツールの出力を入力として使用します。",
)
def call_pr_ai_agent(query: str):
    response = pr_ai_agent.invoke({"input": query})
    return response["output"]


def init_page():
    st.set_page_config(
        page_title="自己PR作成 Agent(multi)",  # ページタイトル
        page_icon="🤗",  # ページアイコン
    )
    st.header("自己PR作成 Agent(multi) 🤗")  # ヘッダーの設定
    st.sidebar.title("Options")  # サイドバーのタイトル


def init_messages():
    # 初期メッセージの設定と会話のクリア機能
    clear_button = st.sidebar.button(
        "Clear Conversation", key="clear"
    )  # 会話をクリアするボタン
    if clear_button or "messages" not in st.session_state:
        # 会話をクリアまたは初期化
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは！なんでも質問をどうぞ！"}
        ]
        st.session_state["memory"] = ConversationBufferWindowMemory(
            return_messages=True,  # メッセージを返す設定
            memory_key="chat_history",  # メモリキーの設定
            k=10,  # 保持するメッセージ数
        )


def create_agent():
    tools = [
        search_ddg,
        fetch_ddg_page,
        call_corporate_analysis_agent,
        call_pr_ai_agent,
    ]  # 使用するツールのリスト
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PR_AI_MULTI_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state["memory"],
        max_iterations=30,
    )


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_agent()

    for msg in st.session_state["memory"].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="株式会社〇〇へ提出する自己PRを作成"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # コールバック関数の設定 (エージェントの動作の可視化用)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

            # エージェントを実行
            response = web_browsing_agent.invoke(
                {"input": prompt}, config=RunnableConfig({"callbacks": [st_cb]})
            )
            st.write(response["output"])


if __name__ == "__main__":
    main()
