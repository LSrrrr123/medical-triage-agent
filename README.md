# 医疗门诊分诊 Agent（PO1）

模拟医院门诊分诊流程的 RAG 智能体：多轮问诊 + 红旗症状识别 + 症状知识库检索 + 结构化分诊建议。

## 功能

- **多轮问诊**：每轮最多 3 个问题，不重复追问，阶段性小结
- **红旗症状优先**：胸痛、腹痛、卒中样等 8 类急危重信号，命中立即建议急诊/120 并停止问诊
- **RAG 检索**：45 张症状卡片入库 Chroma，BGE 中文嵌入，人群/系统过滤
- **结构化输出**：8 项分诊建议（风险等级/科室/处理建议/就医信号/避免事项/复诊/引用/免责）
- **安全边界**：不确诊、不开药、不给剂量；检索失败走安全兜底，不编造依据
- **特殊人群**：儿童/孕妇/老人/女性成人专属卡片与保守分诊策略
- **网页界面**：Streamlit 聊天式交互

## 技术栈

| 层 | 技术 |
|---|---|
| LLM | DeepSeek API（deepseek-chat，OpenAI 兼容协议） |
| 编排 | LangChain（langchain-openai，函数调用） |
| 向量库 | Chroma（本地持久化） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（sentence-transformers） |
| 检索服务 | FastAPI + uvicorn |
| 界面 | Streamlit |
| 语言 | Python 3.13 |

## 文件结构

```
PO1/
├── 医疗门诊分诊Agent需求文档_v1.1.md   # PRD（含验收标准）
├── system_prompt_triage_agent_v1.md    # Agent 系统提示词
├── 验收报告.md                          # 验收结果与简历项目描述
├── symptom_triage_kb_v1.json           # 症状知识库（45 卡）
├── build_index.py                      # 建库脚本（JSON → Chroma）
├── search_tool.py                      # FastAPI 检索服务（8000 端口）
├── agent.py                            # 命令行对话主程序
├── app.py                              # Streamlit 网页界面（8501 端口）
├── test_search.py / test_agent*.py     # 测试脚本
├── requirements.txt / .env / .venv     # 依赖与配置
└── chroma_db/                          # 向量索引（已生成）
```

## 快速开始

```powershell
cd C:\Users\SuGuo\Desktop\PO1

# 1. 建索引（只需首次；知识库更新后重跑）
.\.venv\Scripts\python.exe build_index.py

# 2. 启动检索服务（窗口保持开着）
.\.venv\Scripts\python.exe search_tool.py

# 3. 启动网页界面（另开窗口）
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问 http://localhost:8501 开始对话。命令行版运行 `agent.py`。

## 检索 API

`POST http://127.0.0.1:8000/search`

```json
{
  "query": "成人 右上腹 隐痛 饭后加重 恶心 无发热 红旗",
  "population": "成人",
  "top_k": 5
}
```

返回与主诉最相关的 approved 症状卡片（问诊要点、分诊建议、科室、红旗信号、避免事项）。

## 免责声明

本项目仅用于技术学习与流程演示。知识库内容为示例数据，未经临床审核；所有输出均为分诊参考，不能替代医生面诊、确诊、处方或急诊救治。
