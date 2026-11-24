# 门诊分诊助手 - Streamlit 网页界面
import json
import sys
import httpx
import streamlit as st
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from agent import SYSTEM_PROMPT, TOOLS, SEARCH_URL, load_env, get_llm

load_env()

# ---------- 页面配置 ----------
st.set_page_config(page_title="门诊分诊助手", page_icon="🏥", layout="centered")

st.title("🏥 门诊分诊助手")
st.caption("模拟门诊分诊流程：多轮问诊 + 症状知识库检索（RAG）")

with st.sidebar:
    st.markdown("### 使用说明")
    st.markdown(
        "1. 描述你的主要不适（部位、症状、持续时间）\n"
        "2. 回答助手的追问\n"
        "3. 获取建议科室与就医提醒"
    )
    st.warning("⚠️ 本助手不能替代医生面诊、确诊或处方。出现急危重症状请立即拨打 120。")
    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ---------- LLM（缓存避免重复创建） ----------
@st.cache_resource
def get_llm_cached():
    return get_llm().bind_tools(TOOLS)


# ---------- 会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.ready = True

# ---------- 历史消息展示 ----------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, SystemMessage):
        continue
    elif isinstance(msg, ToolMessage):
        continue
    else:  # AIMessage
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---------- 输入处理 ----------
user_input = st.chat_input("描述你的症状，例如：右上腹隐痛2天，饭后加重……")

if user_input:
    messages = st.session_state.messages
    messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("正在分析……"):
            llm_tools = get_llm_cached()
            for _ in range(3):
                resp = llm_tools.invoke(messages)
                if resp.tool_calls:
                    messages.append(resp)
                    for tc in resp.tool_calls:
                        try:
                            r = httpx.post(SEARCH_URL, json=tc["args"], timeout=15)
                            data = r.json()
                        except Exception as e:
                            data = {"error": f"检索工具调用失败: {e}", "results": []}
                        content = json.dumps(data.get("results", []), ensure_ascii=False)
                        messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
                else:
                    messages.append(resp)
                    break
            st.markdown(resp.content)
