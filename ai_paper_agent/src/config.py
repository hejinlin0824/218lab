import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# =============================================================================
# 1. 环境与路径配置
# =============================================================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
env_path = project_root / '.env'

# 强制加载环境变量 (override=True 确保覆盖系统变量，这对多环境部署很重要)
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    # 允许在没有 .env 的情况下运行（例如通过 docker env 传参）
    print(f"Warning: .env configuration file not found at {env_path}")

# --- 核心目录定义 ---
# DOCS_DIR: 存放原始论文 PDF (公共资源，所有用户共享读取)
DOCS_DIR = project_root / 'docs'

# RES_DIR: 用户会话容器 (根目录)
# 结构: res/ <uuid_session_1> / files...
#       res/ <uuid_session_2> / files...
RES_DIR = project_root / 'res'

# 自动创建根目录
if not DOCS_DIR.exists():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. LLM 模型配置 (单例模式)
# =============================================================================
# 注意：虽然 Agent 是多实例的，但 LLM 客户端对象通常是线程安全的，可以全局复用
# 如果需要支持不同用户使用不同 API Key，这里需要改为工厂模式 (类似 agent_reviewer 的做法)
# 目前为了简化，我们假设后台使用统一的 API Key 服务。

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-pro-preview")
T_SEARCH_API = os.getenv("T_SEARCH_API") 

if not OPENAI_API_KEY:
    # 不抛出致命错误，防止 GUI 启动失败，但在调用时会报错
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# 初始化核心 LLM
llm = ChatOpenAI(
    model=MODEL_NAME, 
    temperature=0.0, # 科研任务建议低温，保持严谨
    api_key=OPENAI_API_KEY,       
    base_url=OPENAI_API_BASE,    
    streaming=True
)

# =============================================================================
# 3. 文件名常量定义
# =============================================================================
# 这些只是文件名字符串，具体路径由 Agent 结合 Session ID 动态生成
FILE_BASE_INFO = "base.md"
FILE_MEMORY = "memory.md"
FILE_INNOV_1 = "innov1.md"
FILE_INNOV_2 = "innov2.md"
FILE_INNOV_3 = "innov3.md"
FILE_FINAL = "final_innov.md"

# Debug Info
print(f"✅ Config loaded. Root RES_DIR: {RES_DIR}")