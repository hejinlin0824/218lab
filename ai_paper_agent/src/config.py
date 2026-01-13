import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 强制加载环境变量
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
env_path = project_root / '.env'

if env_path.exists():
     # 关键修改：override=True 强制覆盖系统环境变量
     load_dotenv(env_path, override=True)
else:
     raise FileNotFoundError(f"严重错误：在 {project_root} 下未找到 .env 配置文件。")

# 2. 路径系统配置 (导出供其他模块使用)
DOCS_DIR = project_root / 'docs'
RES_DIR = project_root / 'res'

if not DOCS_DIR.exists():
     DOCS_DIR.mkdir(parents=True)
if not RES_DIR.exists():
     RES_DIR.mkdir(parents=True)

# 3. LLM 模型配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-pro-preview")
T_SEARCH_API = os.getenv("T_SEARCH_API") # 新增 Tavily Key

if not OPENAI_API_KEY:
     raise ValueError("严重配置错误：环境变量中未找到 OPENAI_API_KEY。")

# --- DEBUG INFO (调试信息) ---
key_prefix = OPENAI_API_KEY[:8] if OPENAI_API_KEY else "None"
print(f"DEBUG: Connecting to {OPENAI_API_BASE}")
print(f"DEBUG: Using Model: {MODEL_NAME}")
print(f"DEBUG: Search API Available: {'Yes' if T_SEARCH_API else 'No'}")
# -------------------------------------------

# 初始化核心 LLM
llm = ChatOpenAI(
     model=MODEL_NAME, 
     temperature=0.0,
     api_key=OPENAI_API_KEY,      
     base_url=OPENAI_API_BASE,    
     streaming=True
)

# 4. 全局常量定义
FILE_BASE_INFO = "base.md"
FILE_MEMORY = "memory.md"
FILE_INNOV_1 = "innov1.md"
FILE_INNOV_2 = "innov2.md"
FILE_INNOV_3 = "innov3.md"
FILE_FINAL = "final_innov.md"

print(f"✅ 配置加载完成。")