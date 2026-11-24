# 检索接口自测：模拟 PRD 验收用例的查询
import httpx

cases = [
    {"query": "成人 右上腹 隐痛 饭后加重 恶心 无发热", "top_k": 3},
    {"query": "胸口压榨样疼痛 大汗 呼吸困难", "top_k": 3},
    {"query": "儿童发热 精神差 8个月", "population": "儿童", "top_k": 3},
    {"query": "怀孕 腹痛 出血", "population": "孕妇", "top_k": 3},
]

for c in cases:
    r = httpx.post("http://127.0.0.1:8000/search", json=c, timeout=10)
    data = r.json()
    print("=" * 40)
    print("查询:", c["query"])
    if "error" in data:
        print("接口错误:", data["error"], data["detail"])
        continue
    for item in data.get("results", []):
        print(f"- {item['symptom']} | 科室: {item['departments'][:2]}")
