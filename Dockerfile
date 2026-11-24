# 医疗门诊分诊 Agent 镜像：检索服务与 Web 界面共用
FROM python:3.13-slim

WORKDIR /app

# 国内镜像加速 HuggingFace 模型下载
ENV HF_ENDPOINT=https://hf-mirror.com \
    PIP_NO_CACHE_DIR=1

# 1. 先装 torch CPU（体积大，单独一层便于缓存）
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 其余依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. 预下载 BGE 中文模型（打进镜像，启动即用）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 4. 代码与数据
COPY build_index.py search_tool.py agent.py app.py eval.py ./
COPY system_prompt_triage_agent_v1.md ./
COPY symptom_triage_kb_v1.json ./
COPY assets ./assets
COPY chroma_db ./chroma_db

EXPOSE 8000 8501

CMD ["python", "search_tool.py"]
