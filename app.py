# 门诊分诊助手 - Streamlit 网页（DeepSeek 风格：左侧历史会话 + 欢迎页 + 底部输入区）
import json
import sys
import time
from datetime import datetime
import httpx
import streamlit as st
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from agent import SYSTEM_PROMPT, TOOLS, SEARCH_URL, load_env, get_llm

load_env()

CONV_DIR = BASE / "conversations"  # 本地会话存档（含健康描述，勿上传）
CONV_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="门诊分诊助手", page_icon="🏥", layout="centered")


# ---------- LLM 缓存（智能搜索开关对应两个实例） ----------
@st.cache_resource
def get_llm_rag(on: bool):
    llm = get_llm()
    return llm.bind_tools(TOOLS) if on else llm


# ---------- 会话存取 ----------
def new_id():
    return str(int(time.time() * 1000))


def save_current():
    """把当前会话写入 conversations/<id>.json"""
    cid = st.session_state.get("current_id")
    if not cid:
        return
    msgs = []
    for m in st.session_state.messages:
        if isinstance(m, SystemMessage):
            continue
        item = {"role": m.type, "content": m.content}
        if getattr(m, "tool_call_id", None):
            item["tool_call_id"] = m.tool_call_id
        if getattr(m, "tool_calls", None):
            item["tool_calls"] = m.tool_calls
        msgs.append(item)
    title = next((m["content"][:20] for m in msgs if m["role"] == "human"), "新对话")
    rec = {"title": title, "updated_at": time.time(), "messages": msgs}
    (CONV_DIR / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def load_conv(cid):
    data = json.loads((CONV_DIR / f"{cid}.json").read_text(encoding="utf-8"))
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in data["messages"]:
        if item["role"] == "human":
            msgs.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            kwargs = {}
            if item.get("tool_calls"):
                kwargs["tool_calls"] = item["tool_calls"]
            msgs.append(AIMessage(content=item["content"], **kwargs))
        elif item["role"] == "tool":
            msgs.append(ToolMessage(content=item["content"], tool_call_id=item.get("tool_call_id", "")))
    st.session_state.messages = msgs
    st.session_state.current_id = cid


def list_convs():
    out = []
    for f in CONV_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append((f.stem, data.get("title", "新对话"), data.get("updated_at", 0)))
        except Exception:
            pass
    out.sort(key=lambda x: -x[2])
    return out


def group_convs(convs):
    """按 今天/昨天/7天内/30天内/更早 分组"""
    groups = [("今天", []), ("昨天", []), ("7天内", []), ("30天内", []), ("更早", [])]
    now = datetime.now().date()
    for cid, title, ts in convs:
        days = (now - datetime.fromtimestamp(ts).date()).days
        idx = 0 if days == 0 else 1 if days == 1 else 2 if days <= 7 else 3 if days <= 30 else 4
        groups[idx][1].append((cid, title))
    return groups


# ---------- 会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
if "current_id" not in st.session_state:
    st.session_state.current_id = None


# ---------- 侧边栏：logo + 开启新对话 + 历史会话 ----------
with st.sidebar:
    st.markdown("### 🏥 门诊分诊助手")
    if st.button("➕ 开启新对话", use_container_width=True):
        save_current()
        st.session_state.current_id = None
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        st.rerun()
    st.divider()
    for label, items in group_convs(list_convs()):
        if not items:
            continue
        st.markdown(f"**{label}**")
        for cid, title in items:
            active = cid == st.session_state.get("current_id")
            if st.button(("▸ " if active else "") + title, key=f"conv_{cid}", use_container_width=True):
                save_current()
                load_conv(cid)
                st.rerun()


# ---------- 主区 ----------
human_count = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])

if human_count == 0:
    # 欢迎页（空状态）
    st.markdown("## 欢迎回来，想从哪里开始？")
    st.caption("我是门诊分诊助手：多轮问诊 + 症状知识库检索。建议仅供参考，不能替代医生面诊。")
    st.divider()
    cols = st.columns(3)
    examples = [
        ("❤️ 胸痛", "胸口压榨痛，出汗，喘不上气"),
        ("🩺 腹痛", "右上腹隐痛2天，饭后加重，恶心"),
        ("👶 儿童发热", "宝宝8个月发烧38.5度，精神不好"),
    ]
    for col, (label, text) in zip(cols, examples):
        if col.button(f"{label}\n\n{text}", use_container_width=True, key=f"ex_{label}"):
            st.session_state.messages.append(HumanMessage(content=text))
            st.session_state.current_id = new_id()
            st.rerun()
else:
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            with st.chat_message("user"):
                st.markdown(m.content)
        elif isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(m.content)


# ---------- 底部输入区 ----------
col1, col2 = st.columns([1, 5])
with col1:
    rag_on = st.toggle("智能搜索", value=True, help="开启后自动检索症状知识库（RAG）")

user_input = st.chat_input("给 门诊分诊助手 发送消息")

if user_input:
    messages = st.session_state.messages
    messages.append(HumanMessage(content=user_input))
    if st.session_state.current_id is None:
        st.session_state.current_id = new_id()
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("正在分析……"):
            llm_tools = get_llm_rag(rag_on)
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
    save_current()
    st.rerun()
