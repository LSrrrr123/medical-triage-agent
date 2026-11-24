# 多轮补充验收：儿童发热、孕妇腹痛、检索失败兜底（走完整链路）
import json
import sys
import httpx
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
import agent as ag
from agent import SYSTEM_PROMPT, TOOLS, load_env, get_llm

load_env()
llm_tools = get_llm().bind_tools(TOOLS)

CASES = [
    ("儿童发热", "http://127.0.0.1:8000/search", [
        "宝宝8个月发烧38.5度，精神不好，不爱喝奶",
        "没有抽搐，但精神很差，叫不太醒，没有皮疹，呼吸还好，尿量比平时少",
    ]),
    ("孕妇腹痛", "http://127.0.0.1:8000/search", [
        "怀孕10周，肚子痛还有一点出血",
        "出血像月经量，肚子一阵阵痛，有点头晕",
    ]),
    ("检索失败兜底", "http://127.0.0.1:9999/search", [
        "头晕3天了，天旋地转，还有点想吐",
        "躺着翻身时明显，每次几秒，没有耳鸣，没有说话不清和肢体无力，没有头痛，血压正常",
    ]),
]


def run_case(title, search_url, turns):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
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
                        r = httpx.post(search_url, json=tc["args"], timeout=10)
                        data = r.json()
                    except Exception as e:
                        data = {"error": f"工具调用失败: {e}", "results": []}
                    content = json.dumps(data.get("results", []), ensure_ascii=False)
                    messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
            else:
                messages.append(resp)
                break
        print("=" * 50)
        print(f"【用例】{title} 第{i+1}轮 | 工具调用: {calls if calls else '无'}")
        print(f"{resp.content}\n")


if __name__ == "__main__":
    for title, url, turns in CASES:
        run_case(title, url, turns)
