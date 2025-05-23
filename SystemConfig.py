import json  # 加到 import 区域

# === 加载 GPU 配置 ===
with open("system_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open("secret_config.json", "r", encoding="utf-8") as f:
    secret_config = json.load(f)

is_use_gpu = config.get("is_use_gpu", True)
is_dev_mode = config.get("is_dev_mode", False)
model_name = config["model_name"]
ollama_url = config["ollama_url"]
local_vllm_url = config["local_vllm_url"]
remote_vllm_url = config["remote_vllm_url"]
qdrant_host = config["qdrant_host"]
qdrant_port = config["qdrant_port"]
default_top_k = config.get("default_top_k", 1)
use_local_llm = config.get("use_local_llm", False)

deepseek_token = secret_config["deepseek_token"]