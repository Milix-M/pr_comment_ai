from os import getenv

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from common.prompt import CUSTOM_SYSTEM_PROMPT
from tools.fetch import fetch_ddg_page
from tools.search import search_ddg

load_dotenv(verbose=True)


def init_page():
    # Streamlitページの基本設定
    st.set_page_config(
        page_title="自己PR作成 Agent",  # ページタイトル
        page_icon="🤗",  # ページアイコン
    )
    st.header("自己PR作成 Agent 🤗")  # ヘッダーの設定
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
    tools = [search_ddg, fetch_ddg_page]  # 使用するツールのリスト
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CUSTOM_SYSTEM_PROMPT),
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

    if prompt := st.chat_input(placeholder="LLMとは？"):
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
