import unicodedata
import json
import uuid
import string
from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
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

import SystemConfig

logger = logging.getLogger(__name__)

# 初始化嵌入模型 & Qdrant 客户端
embedding_model = SentenceTransformer(SystemConfig.model_name)
qdrant_client = QdrantClient(host=SystemConfig.qdrant_host, port=SystemConfig.qdrant_port)

# === FastAPI 初始化 ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ 启动前执行
    logger.info("FastAPI 启动：is_use_gpu: %s", SystemConfig.is_use_gpu)
    logger.info("FastAPI 启动：is_dev_mode: %s", SystemConfig.is_dev_mode)

    yield  # 🟢 应用运行中

    # ✅ 关闭前执行（可选）
    logger.info("FastAPI 即将关闭")
app = FastAPI(title="RAG 工程助手 API", description="结合 Qdrant + Ollama 的智能问答服务", lifespan=lifespan)
# 挂载 static 目录（假设你将 HTML 放在当前目录）
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    top_k: Optional[int] = SystemConfig.default_top_k

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
def retrieve_from_qdrant(query, collection_name="enterprise_knowledge", top_k=SystemConfig.default_top_k):
    query_vec = embedding_model.encode([query], normalize_embeddings=True)[0]
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k
    )
    return [hit.payload for hit in search_result]

# === 构造 Prompt 并调用 Ollama ===
def ask_with_context(user_question, collection_name, model="mistral:7b-instruct", top_k=SystemConfig.default_top_k):
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
    response = requests.post(SystemConfig.ollama_url, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"请求失败: {response.status_code}\n{response.text}"

def is_punctuation(char):

    if not isinstance(char, str) or len(char) != 1:
        # logger.info(f"传入的 char：{repr(char)}，长度：{len(char)}   not isinstance(char, str) or len(char) != 1: {False}")
        return False

    # 放行英文标点符号
    if char in string.punctuation and ord(char) < 128:
        # logger.info(f"传入的 char：{repr(char)}，长度：{len(char)}   char in string.punctuation and ord(char) < 128: {False}")
        return False

    # 括号与引号
    if char in ["（", "）", "(", ")", "{", "}", "[", "]", "'", "\"", "‘", "“"]:
        return False

    # 只算中文标点符号
    # logger.info(f"传入的 char：{repr(char)}，长度：{len(char)}   {unicodedata.category(char).startswith('P')}")
    return unicodedata.category(char).startswith('P')

# === 流式请求 Ollama（支持逐段返回）===
def ask_with_context_stream_ollama_mistral(user_question, collection_name, model="mistral:7b-instruct", top_k=SystemConfig.default_top_k):
    results = retrieve_from_qdrant(user_question, collection_name, top_k=top_k)
    logger.info(f"qdrant results: {results}")

    if not results:
        context = "（未能从知识库中检索到相关资料）"
    else:
        context = "\n".join([
            f"【{r.get('section', r.get('source', '未注明来源'))}】\n{r.get('content', r.get('text', str(r)))}"
            for r in results
        ])

    prompt = f"""你是一个工程智能助手，请结合以下背景资料和用户问题，用**中文**回答用户问题。请严格遵循以下格式与要求：

    1. 回答中必须完全使用中文，**禁止出现任何英文单词或术语**；
        - 即使英文术语（如 depth、spacing、hole 等）出现在背景资料中，也必须翻译为中文；
        - **特别强调：严禁出现“depth”、“spacing”、“hole”等词，即使模型认为更专业也不能使用，必须翻译为“深度”、“间距”、“孔号”等。**
    2. 每句话长度**不能超过10个字符**，包括所有**数字、单位、标点、括号、引号**；
        - 如果一句话超过10个字符，必须拆分为多句；
        - 所有句子必须使用中文标点；
        - 中间句子应使用逗号、顿号结尾，仅最后一句使用句号；
    3. 若有较长的表达，请**分为多句短句**逐条输出；
    4. 不需要解释术语、不要添加注释；
    5. 语言风格应简洁、自然、口语化，适合朗读与语音播放；
    6. 不得使用 markdown 或特殊符号（如 `>`, `-`, `*` 等）；
    7. 你可以数一数每句字符是否超过10个，确保断句合理。

    ⚠️ 注意：这不是建议，是必须遵守的硬性要求。

    下面是一个不合格的回答示例（过长且包含英文）：
    N3的实际邻孔间距在130米处为2626.32毫米，depth为130m。

    请改为如下形式（合格）：
    孔号为N3，  
    邻孔间距在130米处，  
    实际值是2626.32毫米。

    请严格按这种形式回答，不得包含任何英文或特殊符号。

    背景资料：
    {context}

    用户提问：
    {user_question}

    请根据上述要求，输出标准、专业、分句合理的中文回答。
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True  # 开启流模式
    }

    logger.info(f"Request llm server payload: {payload}")

    try:
        with requests.post(SystemConfig.ollama_url, json=payload, stream=True) as response:
            if response.status_code != 200:
                yield f"[错误] 请求失败: {response.status_code}"
                return

            response_buffer = ""
            is_ready_to_send = False
            # 一行一行读取响应内容
            for line in response.iter_lines():
                if line:
                    try:
                        # Ollama 的每一行是 JSON，如 {"response": "部分回答", "done": false}
                        data = json.loads(line.decode("utf-8"))
                        # logger.info(f"Request llm server response: {data}")
                        response = data["response"].strip()
                        if response != "" :
                            if is_punctuation(response):
                                response_buffer += response + "\n"
                                is_ready_to_send = True
                            else:
                                if is_ready_to_send:
                                    is_ready_to_send = False
                                    chunked_response = response_buffer
                                    response_buffer = response
                                    logger.info(f"llm server response: {chunked_response}")
                                    yield chunked_response
                                else:
                                    response_buffer += response
                        if data.get("done"):
                            chunked_response = response_buffer.removesuffix("\n") + "[Heil Hitler!]" + "\n"
                            logger.info(f"llm server response: {chunked_response}")
                            yield chunked_response
                            break
                    except Exception as e:
                        logger.error(f"解析失败: {line} -> {e}")
                        raise e
    except Exception as e:
        logger.error(f"[错误] 连接异常: {e}")
        raise e

# === 流式请求 Ollama（支持逐段返回）===
def ask_with_context_stream_vllm_qwen(user_question, collection_name, top_k=SystemConfig.default_top_k):
    results = retrieve_from_qdrant(user_question, collection_name, top_k=top_k)
    logger.info(f"qdrant results: {results}")

    if not results:
        context = "（未能从知识库中检索到相关资料）"
    else:
        context = "\n".join([
            f"【{r.get('section', r.get('source', '未注明来源'))}】\n{r.get('content', r.get('text', str(r)))}"
            for r in results
        ])

    prompt = f"""你是一个工程智能助手，请结合以下背景资料和用户问题，用**中文**回答用户问题。请严格遵循以下格式与要求：

    1. 回答中必须完全使用中文，**禁止出现任何英文单词或术语**；
        - 即使英文术语（如 depth、spacing、hole 等）出现在背景资料中，也必须翻译为中文；
        - **特别强调：严禁出现“depth”、“spacing”、“hole”等词，即使模型认为更专业也不能使用，必须翻译为“深度”、“间距”、“孔号”等。**
    2. 每句话长度**不能超过10个字符**，包括所有**数字、单位、标点、括号、引号**；
        - 如果一句话超过10个字符，必须拆分为多句；
        - 所有句子必须使用中文标点；
        - 中间句子应使用逗号、顿号结尾，仅最后一句使用句号；
    3. 若有较长的表达，请**分为多句短句**逐条输出；
    4. 不需要解释术语、不要添加注释；
    5. 语言风格应简洁、自然、口语化，适合朗读与语音播放；
    6. 不得使用 markdown 或特殊符号（如 `>`, `-`, `*` 等）；
    7. 你可以数一数每句字符是否超过10个，确保断句合理。

    ⚠️ 注意：这不是建议，是必须遵守的硬性要求。

    下面是一个不合格的回答示例（过长且包含英文）：
    N3的实际邻孔间距在130米处为2626.32毫米，depth为130m。

    请改为如下形式（合格）：
    孔号为N3，  
    邻孔间距在130米处，  
    实际值是2626.32毫米。

    请严格按这种形式回答，不得包含任何英文或特殊符号。

    背景资料：
    {context}

    用户提问：
    {user_question}

    请根据上述要求，输出标准、专业、分句合理的中文回答。
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True  # 开启流模式
    }

    logger.info(f"Request llm server payload: {payload}")

    try:
        with requests.post(SystemConfig.ollama_url, json=payload, stream=True) as response:
            if response.status_code != 200:
                yield f"[错误] 请求失败: {response.status_code}"
                return

            response_buffer = ""
            is_ready_to_send = False
            # 一行一行读取响应内容
            for line in response.iter_lines():
                if line:
                    try:
                        # Ollama 的每一行是 JSON，如 {"response": "部分回答", "done": false}
                        data = json.loads(line.decode("utf-8"))
                        # logger.info(f"Request llm server response: {data}")
                        response = data["response"].strip()
                        if response != "" :
                            if is_punctuation(response):
                                response_buffer += response + "\n"
                                is_ready_to_send = True
                            else:
                                if is_ready_to_send:
                                    is_ready_to_send = False
                                    chunked_response = response_buffer
                                    response_buffer = response
                                    logger.info(f"llm server response: {chunked_response}")
                                    yield chunked_response
                                else:
                                    response_buffer += response
                        if data.get("done"):
                            chunked_response = response_buffer.removesuffix("\n") + "[Heil Hitler!]" + "\n"
                            logger.info(f"llm server response: {chunked_response}")
                            yield chunked_response
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
    # default use qwen llm
    return StreamingResponse(
        ask_with_context_stream_vllm_qwen(
            req.question,
            collection,
            top_k=req.top_k),
        media_type="text/plain",
        status_code=200,
        headers={"X-Accel-Buffering": "no"}
    )




if __name__ == "__main__":
    import uvicorn

    if SystemConfig.is_dev_mode:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=SystemConfig.is_dev_mode,
            log_config="log_config.yml"
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=SystemConfig.is_dev_mode,
            log_config="log_config.yml"
        )