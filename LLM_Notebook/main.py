import streamlit as st
from streamlit_quill import st_quill
from markdownify import markdownify as md
import time
from pathlib import Path

# 导入自定义配置与工具 (确保 config.py 和 utils.py 已按之前代码保存)
from config import APP_TITLE, EXCLUDE_FILES, get_user_context
from utils import (
    list_markdown_files, 
    read_file_content, 
    save_file_content, 
    delete_file, 
    validate_path_security
)

# =============================================================================
# 1. 页面基础配置
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS：美化编辑器、侧边栏及按钮样式
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    .stRadio > label { font-weight: bold; color: #6366f1; }
    .user-info {
        padding: 10px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 20px;
    }
    /* 针对 type="secondary" 的按钮进行红色高亮，模拟危险区域 */
    .stButton button[kind="secondary"] {
        color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. 身份与路径锁定 (实名制隔离)
# =============================================================================
current_user, user_root = get_user_context()

# =============================================================================
# 3. 侧边栏：增、删、查 逻辑
# =============================================================================
with st.sidebar:
    st.title("🧪 218 实验室笔记")
    st.markdown(f"""
        <div class="user-info">
            <small style="color: #94a3b8;">当前研究员</small><br>
            <strong>{current_user}</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # --- 查 (List) ---
    st.subheader("📂 笔记库")
    files = list_markdown_files(user_root, EXCLUDE_FILES)
    
    # 使用 selectbox 快速切换当前编辑的文件
    if not files:
        st.info("当前目录下暂无笔记")
        selected_file_rel = None
    else:
        # 默认选中第一个
        selected_file_rel = st.selectbox(
            "选择要查看或编辑的笔记",
            options=files,
            index=0
        )

    st.divider()

    # --- 增 (Create) ---
    with st.expander("✨ 新建科研笔记", expanded=False):
        new_note_name = st.text_input("笔记名称", placeholder="例如: 实验思路_v1")
        if st.button("确认创建"):
            if new_note_name:
                # 规范化文件名
                safe_name = new_note_name.strip().replace(" ", "_")
                if not safe_name.endswith(".md"):
                    safe_name += ".md"
                
                target_new_path = user_root / safe_name
                # 初始化一个简单的标题
                initial_md = f"# {new_note_name}\n在此输入您的研究内容..."
                success, msg = save_file_content(target_new_path, initial_md)
                if success:
                    st.success("创建成功！")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

    # --- 删 (Delete) ---
    if selected_file_rel:
        st.divider()
        st.subheader("⚠️ 危险区域")
        # 【修正点】：将 kind="secondary" 修改为 type="secondary"
        if st.button("🗑️ 永久删除当前笔记", type="secondary"):
            path_to_del = user_root / selected_file_rel
            if validate_path_security(path_to_del, user_root):
                success, msg = delete_file(path_to_del)
                if success:
                    st.toast(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

# =============================================================================
# 4. 主界面：改 (Update) 与 实时预览
# =============================================================================
if selected_file_rel:
    full_file_path = user_root / selected_file_rel
    
    # 安全拦截：防止非法路径访问
    if not validate_path_security(full_file_path, user_root):
        st.error("权限拒绝：无法访问该路径。")
        st.stop()

    # 1. 读取当前物理文件内容作为编辑器初始值
    current_md_content = read_file_content(full_file_path)

    st.subheader(f"📝 正在编辑: {selected_file_rel}")

    # 2. 富文本编辑器 (WYSIWYG)
    content_html = st_quill(
        value=current_md_content,
        placeholder="撰写您的科研灵感、公式或数据分析...",
        html=True,
        key="quill_editor",
        toolbar=[
            ["bold", "italic", "underline", "strike"],
            [{"header": 1}, {"header": 2}],
            [{"list": "ordered"}, {"list": "bullet"}],
            ["link", "image", "code-block"],
            ["clean"],
        ]
    )

    # 3. 实时转换逻辑：HTML -> Markdown (存盘用)
    converted_markdown = md(content_html, heading_style="ATX")

    # 4. 操作按钮
    col_save, col_view = st.columns([1, 1])
    
    with col_save:
        if st.button("💾 存档并同步 (Sync to AI Agent)", type="primary"):
            # 保存到物理磁盘
            success, msg = save_file_content(full_file_path, converted_markdown)
            if success:
                st.balloons()
                st.success(f"{msg}！现在可以在 8218 端口同步此内容了。")
            else:
                st.error(msg)

    st.divider()

    # 5. 实时预览区域 (编译后的排版效果)
    st.markdown("### 👁️ 最终排版预览 (Rendered Preview)")
    with st.container(border=True):
        st.markdown(converted_markdown)
        
    # 可选：查看源码（用于调试）
    with st.expander("🔍 查看转换后的 Markdown 源码"):
        st.code(converted_markdown, language="markdown")

else:
    # 欢迎页逻辑
    st.markdown(f"""
        <div style='text-align: center; margin-top: 150px;'>
            <h1 style='color: #6366f1;'>218 Lab 科研工作空间</h1>
            <p style='color: #94a3b8; font-size: 1.2rem;'>研究员 <strong>{current_user}</strong>，欢迎回来。</p>
            <p style='color: #64748b;'>请在左侧侧边栏选择一个笔记文件，或创建新的研究记录。</p>
        </div>
    """, unsafe_allow_html=True)

# 页脚
st.divider()
st.caption(f"© 2026 218 Lab Center | 实时互通模式已开启 | 存储分区: {current_user}")