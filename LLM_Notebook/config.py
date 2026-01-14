import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量（如果存在）
load_dotenv()

# =============================================================================
# 1. 共享路径配置 (实现与 ai_paper_agent 的底层互通)
# =============================================================================
# 逻辑：
# 当前文件：~/home/218lab/LLM_Notebook/config.py
# 目标目录：~/home/218lab/ai_paper_agent/res

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 核心：直接指向老项目的资源目录，确保操作的是同一块物理硬盘
SHARED_RES_DIR = PROJECT_ROOT / "ai_paper_agent" / "res"

# =============================================================================
# 2. 身份识别与隔离逻辑
# =============================================================================
def get_user_context():
    """
    从 URL 参数获取当前操作的研究员身份。
    实现与导航页 (index.html) 及 AI Agent (8218端口) 强一致的实名隔离。
    """
    # 获取 URL 中的 ?user=xxx 参数
    url_params = st.query_params
    current_user = url_params.get("user", "admin")  # 默认回退到 admin
    
    # 锁定该研究员的专属根目录 (例如: res/hejinlin)
    user_root_path = SHARED_RES_DIR / current_user
    
    # 确保研究员主目录存在，不存在则自动创建
    if not user_root_path.exists():
        user_root_path.mkdir(parents=True, exist_ok=True)
        
    return current_user, user_root_path

# =============================================================================
# 3. 笔记本全局常量
# =============================================================================
APP_TITLE = "218 Lab | Researcher Notebook"
# 排除某些系统文件不显示在笔记本目录树中
EXCLUDE_FILES = ["faiss_index", "memory.md", "figures"]