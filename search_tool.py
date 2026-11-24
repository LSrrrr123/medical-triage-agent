# 检索服务：FastAPI 封装 search_symptom_triage_kb 工具
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel

BASE = Path(__file__).parent
DB_DIR = BASE / "chroma_db"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # BGE 官方推荐的查询前缀

app = FastAPI(title="门诊分诊症状-处理建议库检索工具")
client = chromadb.PersistentClient(path=str(DB_DIR))
col = client.get_collection("symptom_triage")
model = SentenceTransformer(EMB_MODEL)


class SearchReq(BaseModel):
    query: str                # 症状、人群、伴随症状、风险词
    population: str = None    # 成人/儿童/孕妇/老人/女性成人
    system: str = None        # 系统分类
    top_k: int = 5            # 默认 5


@app.post("/search")
def search(req: SearchReq):
    # 硬过滤：仅 approved；选填条件叠加（PRD 6.5）。Chroma 多条件用 $and
    filters = [{"review_status": "approved"}]
    if req.population:
        filters.append({"population": req.population})
    if req.system:
        filters.append({"system": req.system})
    where = filters[0] if len(filters) == 1 else {"$and": filters}

    q_emb = model.encode(QUERY_PREFIX + req.query, normalize_embeddings=True)
    try:
        res = col.query(query_embeddings=[q_emb.tolist()],
                        n_results=req.top_k, where=where)
    except Exception as e:
        # 失败不编造，返回明确错误，由 Agent 走兜底（PRD 8.4）
        return {"error": "检索失败", "detail": str(e), "results": []}

    results = []
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    for i, m in enumerate(metas):
        if m is None:
            continue
        results.append({
            "symptom_id": ids[i],
            "symptom": m["symptom"],
            "content": m["content"],
            "triage_advice": m["triage_advice"],
            "departments": json.loads(m["departments"]),
            "red_flags": json.loads(m["red_flags"]),
            "avoid": json.loads(m["avoid"]),
            "review_status": m["review_status"],
        })
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
