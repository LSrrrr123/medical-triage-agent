# 建库脚本：读知识库 JSON → 拼接检索文本 → 嵌入 → 写入 Chroma
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
KB_FILE = BASE / "symptom_triage_kb_v1.json"
DB_DIR = BASE / "chroma_db"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文小模型，效果好、下载快


def build_doc(card):
    """按 PRD 6.3 拼接检索文本"""
    parts = [
        card["symptom"],
        " ".join(card["aliases"]),
        card["content"],
        card["triage_advice"],
        " ".join(card["departments"]),
        " ".join(card["red_flags"]),
        " ".join(card["avoid"]),
    ]
    return " ".join(parts)


def main():
    kb = json.loads(KB_FILE.read_text(encoding="utf-8"))
    cards = [c for c in kb["cards"] if c["review_status"] == "approved"]

    model = SentenceTransformer(EMB_MODEL)  # 有 GPU 自动用 GPU
    docs = [build_doc(c) for c in cards]
    embeddings = model.encode(docs, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection("symptom_triage")  # 重建库，保证幂等
    except Exception:
        pass
    # 嵌入已归一化，默认 L2 距离与余弦相似度等价，无需额外配置
    col = client.create_collection("symptom_triage")

    ids, metadatas = [], []
    for c in cards:
        ids.append(c["symptom_id"])
        # Chroma metadata 只支持基础类型，数组转 JSON 字符串存储
        metadatas.append({
            "symptom": c["symptom"],
            "population": c["population"],
            "system": c["system"],
            "departments": json.dumps(c["departments"], ensure_ascii=False),
            "red_flags": json.dumps(c["red_flags"], ensure_ascii=False),
            "avoid": json.dumps(c["avoid"], ensure_ascii=False),
            "content": c["content"],
            "triage_advice": c["triage_advice"],
            "review_status": c["review_status"],
            "version": c["version"],
        })

    col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    print(f"入库完成：{len(ids)} 张卡片 -> {DB_DIR}")


if __name__ == "__main__":
    main()
