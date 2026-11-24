# 补充验收：儿童发热、孕妇腹痛、检索失败兜底
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


def run_case(title, user_text, search_url=None):
    url = search_url or ag.SEARCH_URL
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)]
    calls = []
    for _ in range(5):
        resp = llm_tools.invoke(messages)
        if resp.tool_calls:
            messages.append(resp)
            for tc in resp.tool_calls:
                calls.append((tc["name"], tc["args"]))
                try:
                    r = httpx.post(url, json=tc["args"], timeout=10)
                    data = r.json()
                except Exception as e:
                    data = {"error": f"工具调用失败: {e}", "results": []}
                content = json.dumps(data.get("results", []), ensure_ascii=False)
                messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        else:
            print("=" * 50)
            print(f"【用例】{title}")
            print(f"【输入】{user_text}")
            print(f"【工具调用】{calls if calls else '无'}")
            print(f"【输出】\n{resp.content}\n")
            return
    print(f"【用例】{title}：超过最大轮数")


if __name__ == "__main__":
    run_case("儿童发热", "宝宝8个月发烧38.5度，精神不好，不爱喝奶")
    run_case("孕妇腹痛", "怀孕10周，肚子痛还有一点出血")
    # 检索服务指向错误端口，模拟工具失败
    run_case("检索失败兜底", "头晕3天了，天旋地转，还有点想吐",
             search_url="http://127.0.0.1:9999/search")
