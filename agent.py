# 分诊 Agent 主程序：系统提示词 + DeepSeek LLM + 本地 RAG 检索工具
import json
import os
import httpx
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

BASE = Path(__file__).parent
PROMPT_FILE = BASE / "system_prompt_triage_agent_v1.md"
SEARCH_URL = os.environ.get("SEARCH_URL", "http://127.0.0.1:8000/search")  # Docker 内可覆盖


# ---------- 0. 读取项目 .env 文件（每行一条 key=value） ----------
def load_env():
    env_file = BASE / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()

# ---------- 1. 系统提示词 ----------
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

# ---------- 2. DeepSeek（OpenAI 兼容接口） ----------
def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )

# ---------- 3. 检索工具定义（PRD 第8章） ----------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_symptom_triage_kb",
        "description": "检索门诊分诊症状知识库，返回与主诉最相关的症状卡片（问诊要点、分诊处理建议、建议科室、红旗信号、避免事项）。只用于辅助分诊，不得用于确诊或处方。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "主诉部位+症状性质+持续时间+伴随症状+特殊人群+风险词，如：成人 右上腹 隐痛 饭后加重 恶心 无发热 分诊处理建议 红旗",
                },
                "population": {
                    "type": "string",
                    "enum": ["成人", "儿童", "孕妇", "老人", "女性成人"],
                    "description": "选填，目标人群",
                },
                "system": {"type": "string", "description": "选填，系统分类"},
                "top_k": {"type": "integer", "description": "选填，返回条数，默认5"},
            },
            "required": ["query"],
        },
    },
}]

llm_with_tools = None  # 在 chat() 中创建，避免无 Key 时启动报错


# ---------- 4. 工具执行：调本地检索服务 ----------
def call_search(args: dict) -> str:
    try:
        r = httpx.post(SEARCH_URL, json=args, timeout=15)
        data = r.json()
    except Exception as e:
        # 服务不可用也返回明确错误，由 LLM 按提示词走兜底
        return json.dumps({"error": f"检索工具调用失败: {e}", "results": []}, ensure_ascii=False)
    if "error" in data:
        return json.dumps({"error": data["error"], "results": []}, ensure_ascii=False)
    return json.dumps(data["results"], ensure_ascii=False)


# ---------- 5. 对话主循环 ----------
def chat():
    llm_with_tools = get_llm().bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    print("门诊分诊助手已启动（输入 exit 退出）\n")
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() == "exit":
            break
        messages.append(HumanMessage(content=user_input))

        # 单轮内 LLM 可能多次调用工具，最多 3 次防死循环
        for _ in range(3):
            # 流式生成；一旦出现工具调用，改非流式获取完整参数
            full = None
            call_tool = False
            for chunk in llm_with_tools.stream(messages):
                full = chunk if full is None else full + chunk
                if chunk.tool_call_chunks:
                    call_tool = True
                    break
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            if call_tool:
                resp = llm_with_tools.invoke(messages)
                messages.append(resp)
                for tc in resp.tool_calls:
                    result = call_search(tc["args"])
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                continue
            messages.append(full)
            print("\n")
            break
        else:
            print("\n助手: 检索次数过多，请先回答我上面的问题，我们再继续。\n")


if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("请先设置 API Key（二选一）：")
        print("  1. 在 PO1 文件夹建 .env 文件，写入：DEEPSEEK_API_KEY=sk-xxx")
        print("  2. 设置环境变量：$env:DEEPSEEK_API_KEY = 'sk-xxx'")
    else:
        chat()
