import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# 1. 环境与路径配置
# =============================================================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
env_path = project_root / '.env'

# 加载环境变量 (主要用于 T_SEARCH_API 等非 LLM 的辅助工具，或者作为默认值)
if env_path.exists():
    load_dotenv(env_path, override=True)

# --- 核心目录定义 ---
# DOCS_DIR: 存放原始论文 PDF (公共资源，所有用户共享读取)
DOCS_DIR = project_root / 'docs'

# RES_DIR: 用户会话容器 (根目录)
# 结构: res/ <username> / files...
RES_DIR = project_root / 'res'

# 自动创建根目录
if not DOCS_DIR.exists():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
if not RES_DIR.exists():
    RES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. 文件名常量定义
# =============================================================================
# 这些只是文件名字符串，具体路径由 Agent 结合 Session ID (用户名) 动态生成
FILE_BASE_INFO = "base.md"
FILE_MEMORY = "memory.md"
FILE_INNOV_1 = "innov1.md"
FILE_INNOV_2 = "innov2.md"
FILE_INNOV_3 = "innov3.md"
FILE_FINAL = "final_innov.md"

# Debug Info
print(f"✅ Config loaded. Root RES_DIR: {RES_DIR}")