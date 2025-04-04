import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct

# === 配置部分 ===
MODEL_NAME = "BAAI/bge-large-zh"
OLLAMA_URL = "http://localhost:11434/api/generate"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "enterprise_knowledge"

# 初始化嵌入模型 & Qdrant 客户端
embedding_model = SentenceTransformer(MODEL_NAME)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# === Step 1: 查询 Qdrant ===
def retrieve_from_qdrant(query, top_k=3):
    query_vec = embedding_model.encode([query], normalize_embeddings=True)[0]
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k
    )
    return [hit.payload for hit in search_result]


# === Step 2: 构造 Prompt 并调用 Ollama 生成回答 ===
def ask_with_context(user_question, model="llama3", top_k=3):
    # 检索相关内容
    results = retrieve_from_qdrant(user_question, top_k=top_k)
    if not results:
        context = "（未能从知识库中检索到相关资料）"
    else:
        context = "\n".join([f"【{r['section']}】\n{r['content']}" for r in results])

    # 构造提示词
    prompt = f"""你是一个工程智能助手，请结合以下背景知识回答用户问题。

已知资料：
{context}

问题：{user_question}
请用专业、简明的方式回答。
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"请求失败: {response.status_code}\n{response.text}"

def resolve_collection(user_question: str):
    # 简单关键词 → Collection 映射规则
    mapping = {
        "冻结AI": "enterprise_knowledge",
        "吉安": "enterprise_knowledge",
        "静态数据": "enterprise_knowledge",
        "终孔偏斜": "terminal_deviation_data",
        "邻孔间距": "neighbor_spacing_data",
        "钻孔偏斜": "hole_deviation_data",
    }

    for keyword, collection in mapping.items():
        if keyword in user_question:
            return collection
    return "enterprise_knowledge"  # 默认 fallback


# === 示例 ===
if __name__ == "__main__":
    query = "东欢坨项目的井筒设计深度是多少？"
    collection_name = resolve_collection(user_question)
    answer = ask_with_context(query, model="mistral:7b-instruct")
    print("回答：", answer)
