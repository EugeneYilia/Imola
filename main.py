import json
import uuid
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import logging
from colorlog import ColoredFormatter

LOG_FORMAT = "%(log_color)s%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

formatter = ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT, log_colors=LOG_COLORS)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)

# === 配置部分 ===
MODEL_NAME = "BAAI/bge-large-zh"
OLLAMA_URL = "http://localhost:11434/api/generate"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# 初始化嵌入模型 & Qdrant 客户端
embedding_model = SentenceTransformer(MODEL_NAME)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# === FastAPI 初始化 ===
app = FastAPI(title="RAG 工程助手 API", description="结合 Qdrant + Ollama 的智能问答服务")
# 挂载 static 目录（假设你将 HTML 放在当前目录）
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定前端来源 ['http://localhost:3000']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置访问根路径时返回 index.html
@app.get("/")
def serve_home():
    return FileResponse("index.html")

# === 请求体模型 ===
class QuestionRequest(BaseModel):
    question: str
    model: Optional[str] = "mistral:7b-instruct"
    top_k: Optional[int] = 3

# === collection 映射逻辑 ===
def resolve_collection(user_question: str):
    mapping = {
        "大模型需求": "enterprise_knowledge",
        "吉安": "enterprise_knowledge",
        "静态数据": "enterprise_knowledge",
        "终孔偏斜": "terminal_deviation_data",
        "邻孔间距": "neighbor_spacing_data",
        "钻孔偏斜": "hole_deviation_data",
    }
    for keyword, collection in mapping.items():
        if keyword in user_question:
            return collection
    return "enterprise_knowledge"

# === Qdrant 检索 ===
def retrieve_from_qdrant(query, collection_name="enterprise_knowledge", top_k=3):
    query_vec = embedding_model.encode([query], normalize_embeddings=True)[0]
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k
    )
    return [hit.payload for hit in search_result]

# === 构造 Prompt 并调用 Ollama ===
def ask_with_context(user_question, collection_name, model="mistral:7b-instruct", top_k=3):
    results = retrieve_from_qdrant(user_question, collection_name, top_k=top_k)
    logger.info(f"qdrant results: {results}")
    if not results:
        context = "（未能从知识库中检索到相关资料）"
    else:
        context = "\n".join([
            f"【{r.get('section', r.get('source', '未注明来源'))}】\n{r.get('content', r.get('text', str(r)))}"
            for r in results
        ])

    prompt = f"""你是一个工程智能助手，请结合以下背景知识回答用户问题：\n\n已知资料：\n{context}\n\n问题：{user_question}\n请用专业、简明的方式回答。"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    logger.info(f"Request llm server payload: {payload}")
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"请求失败: {response.status_code}\n{response.text}"

# === 流式请求 Ollama（支持逐段返回）===
def ask_with_context_stream(user_question, collection_name, model="mistral:7b-instruct", top_k=3):
    results = retrieve_from_qdrant(user_question, collection_name, top_k=top_k)
    logger.info(f"qdrant results: {results}")

    if not results:
        context = "（未能从知识库中检索到相关资料）"
    else:
        context = "\n".join([
            f"【{r.get('section', r.get('source', '未注明来源'))}】\n{r.get('content', r.get('text', str(r)))}"
            for r in results
        ])

    prompt = f"""你是一个工程智能助手，请结合以下背景知识回答用户问题：\n\n已知资料：\n{context}\n\n问题：{user_question}\n请用专业、简明的方式回答。"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True  # 开启流模式
    }

    logger.info(f"Request llm server payload: {payload}")

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                yield f"[错误] 请求失败: {response.status_code}"
                return

            # 一行一行读取响应内容
            for line in response.iter_lines():
                if line:
                    try:
                        # Ollama 的每一行是 JSON，如 {"response": "部分回答", "done": false}
                        data = json.loads(line.decode("utf-8"))
                        logger.info(f"Request llm server response: {data}")
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            yield "[Heil Hitler!]"
                            break
                    except Exception as e:
                        logger.error(f"解析失败: {line} -> {e}")
                        raise e
    except Exception as e:
        logger.error(f"[错误] 连接异常: {e}")
        raise e

# === API 路由 ===
@app.post("/ask")
def rag_qa(req: QuestionRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"[ASK 接收] {request_id}  问题: {req}")

    collection = resolve_collection(req.question)
    return StreamingResponse(
        ask_with_context_stream(req.question, collection, model=req.model, top_k=req.top_k),
        media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config="log_config.yml"
    )
