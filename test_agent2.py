# 多轮对话验证：腹痛用例完整链路（追问 → 检索 → 8项建议）
import json
import sys
import httpx
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from agent import SYSTEM_PROMPT, TOOLS, SEARCH_URL, load_env, get_llm

load_env()
llm_tools = get_llm().bind_tools(TOOLS)

messages = [SystemMessage(content=SYSTEM_PROMPT)]
turns = [
    "右上腹隐痛2天，饭后加重，恶心，没有发热",
    "不放射，隐痛能忍，没有呕吐，皮肤眼睛不发黄，不是怀孕，以前有胆囊结石",
]

for i, t in enumerate(turns):
    messages.append(HumanMessage(content=t))
    calls = []
    for _ in range(3):
        resp = llm_tools.invoke(messages)
        if resp.tool_calls:
            messages.append(resp)
            for tc in resp.tool_calls:
                calls.append((tc["name"], tc["args"]))
                try:
                    r = httpx.post(SEARCH_URL, json=tc["args"], timeout=15)
                    data = r.json()
                except Exception as e:
                    data = {"error": f"工具调用失败: {e}", "results": []}
                content = json.dumps(data.get("results", []), ensure_ascii=False)
                messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        else:
            messages.append(resp)
            break
    print(f"===== 第{i+1}轮 =====")
    print(f"工具调用: {calls if calls else '无'}")
    print(f"{resp.content}\n")
