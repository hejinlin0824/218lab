import streamlit as st
import os
import time
from pathlib import Path

# 引入核心模块
from src.agent import ResearchAgent
from src.config import (
    DOCS_DIR, RES_DIR, 
    FILE_BASE_INFO, FILE_MEMORY, FILE_FINAL,
    FILE_INNOV_1, FILE_INNOV_2, FILE_INNOV_3
)
from src.prompts import PromptManager

# 引入 Streamlit 官方的回调处理器
from langchain_community.callbacks import StreamlitCallbackHandler

# =============================================================================
# 🔴 关键配置：请在这里填入您的服务器 IP
# =============================================================================
SERVER_PUBLIC_IP = "localhost" 

# =============================================================================
# 0. 页面基础配置 (原生风格)
# =============================================================================
st.set_page_config(
    page_title="218 Lab | AI Research Agent", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. 身份识别与配置
# =============================================================================
if "user_session_id" not in st.session_state:
    query_params = st.query_params
    url_user = query_params.get("user")
    
    if url_user:
        st.session_state.user_session_id = url_user
    else:
        st.session_state.user_session_id = "admin"

# 锁定当前用户的物理目录
USER_RES_DIR = RES_DIR / st.session_state.user_session_id
if not USER_RES_DIR.exists():
    USER_RES_DIR.mkdir(parents=True, exist_ok=True)

# 笔记本端口映射
NOTEBOOK_PORTS = {
    "hejinlin": 8002,
    "zhaoyixin": 8003,
    "admin": 8002
}

# =============================================================================
# 2. 核心工具函数 (提前定义以便侧边栏调用)
# =============================================================================
def check_milestone(filename):
    return (USER_RES_DIR / filename).exists()

def read_file_content(filename):
    path = USER_RES_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def clean_project_files(scope="partial"):
    """清理项目文件：partial仅清除创新点，full清除所有"""
    files_to_remove = [FILE_MEMORY, FILE_INNOV_1, FILE_INNOV_2, FILE_INNOV_3, FILE_FINAL, "total.md"]
    
    if scope == "full":
        files_to_remove.append(FILE_BASE_INFO)
        figures_dir = USER_RES_DIR / "figures"
        if figures_dir.exists():
            try:
                for item in figures_dir.iterdir():
                    if item.is_file(): item.unlink()
            except Exception: pass

    for filename in files_to_remove:
        path = USER_RES_DIR / filename
        if path.exists():
            try:
                os.remove(path)
            except Exception: pass

def merge_final_report():
    target_files = [FILE_INNOV_1, FILE_INNOV_2, FILE_INNOV_3, FILE_FINAL]
    total_content = ["# Final Research Proposal\n", f"> **Researcher**: {st.session_state.user_session_id}\n", f"> **Date**: {time.strftime('%Y-%m-%d')}\n", "---\n"]
    for fname in target_files:
        content = read_file_content(fname)
        if content:
            total_content.append(f"\n\n---\n\n") 
            total_content.append(content)
    with open(USER_RES_DIR / "total.md", 'w', encoding='utf-8') as f:
        f.write("".join(total_content))

# 模态弹窗预览文件
@st.dialog("📄 文件预览")
def show_file_content(filename, content):
    st.caption(f"File: {filename}")
    st.markdown(content)

# =============================================================================
# 3. 侧边栏：配置、导航与进度管理
# =============================================================================
with st.sidebar:
    st.title("🎓 218 科研助手")
    st.caption(f"User: {st.session_state.user_session_id}")
    
    # --- A. 模型配置 ---
    with st.expander("⚙️ 模型配置", expanded=True):
        base_url_options = {
            "OpenAI (官方)": "https://api.openai.com/v1",
            "DeepSeek (深度求索)": "https://api.deepseek.com/v1",
            "Aihubmix (中转)": "https://aihubmix.com/v1",
            "自定义 (Custom)": "custom"
        }
        
        selected_provider = st.selectbox("服务商", list(base_url_options.keys()))
        
        if selected_provider == "自定义 (Custom)":
            user_base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
        else:
            user_base_url = st.text_input("Base URL", value=base_url_options[selected_provider])

        user_api_key = st.text_input("API Key", type="password", placeholder="sk-...")

        default_models = {"OpenAI (官方)": "gpt-4o", "DeepSeek (深度求索)": "deepseek-chat", "Aihubmix (中转)": "gemini-1.5-pro-latest", "自定义 (Custom)": "gpt-4o"}
        user_model_name = st.text_input("模型名称", value=default_models.get(selected_provider, "gpt-4o"))

        config_ready = bool(user_api_key and user_base_url and user_model_name)
        if config_ready:
            st.success("✅ 已连接")
        else:
            st.warning("⚠️ 需配置 Key")

    st.divider()

    # --- B. 笔记本跳转 ---
    current_user = st.session_state.user_session_id
    user_port = NOTEBOOK_PORTS.get(current_user, "0000")
    if user_port != "0000":
        final_url = f"http://{SERVER_PUBLIC_IP}:{user_port}"
        st.link_button("📓 打开专属笔记本", final_url, use_container_width=True)
        if st.button("🔄 同步向量记忆", disabled=not config_ready, use_container_width=True):
            if "agent" in st.session_state:
                with st.spinner("Indexing..."):
                    res = st.session_state.agent.sync_knowledge_base()
                    st.toast(res)
            else:
                st.error("请先初始化")
    
    st.divider()

    # --- C. 进度可视化 (复原功能) ---
    st.subheader("📊 研究进度")
    
    # 进度状态推断 (需要在渲染前更新一次 session_state.phase)
    if "phase" not in st.session_state: st.session_state.phase = "init"
    
    def render_step_status(label, filename, associated_phases):
        col1, col2 = st.columns([0.8, 0.2])
        is_completed = check_milestone(filename)
        is_doing = (st.session_state.phase in associated_phases) and not is_completed
        
        with col1:
            if is_completed:
                st.markdown(f"✅ ~~{label}~~")
            elif is_doing:
                st.markdown(f"**🔄 :blue[{label}]**")
            else:
                st.markdown(f"⚪ <span style='color:grey'>{label}</span>", unsafe_allow_html=True)
        
        with col2:
            if is_completed:
                # 使用唯一 Key 防止冲突
                if st.button("📄", key=f"view_{filename}", help="查看文件"):
                    content = read_file_content(filename)
                    if content: show_file_content(filename, content)

    render_step_status("阅读基准 (Base)", FILE_BASE_INFO, ["read"])
    render_step_status("创新点 1 (Innov1)", FILE_INNOV_1, ["innov1"])
    render_step_status("创新点 2 (Innov2)", FILE_INNOV_2, ["innov2"])
    render_step_status("创新点 3 (Innov3)", FILE_INNOV_3, ["innov3"])
    render_step_status("实验设计 (Final)", FILE_FINAL, ["final"])

    st.divider()

    # --- D. 重置选项 (复原功能) ---
    with st.expander("⚠️ 重置/危险区", expanded=False):
        if st.button("🔙 重置创新点 (保留Base)", use_container_width=True):
            clean_project_files("partial")
            st.session_state.clear()
            st.rerun()
            
        if st.button("🆕 彻底重置 (新课题)", type="primary", use_container_width=True):
            clean_project_files("full")
            st.session_state.clear()
            st.rerun()

    # --- E. 论文上传 ---
    st.divider()
    uploaded_file = st.file_uploader("📂 上传 PDF", type=["pdf"])
    if uploaded_file:
        save_path = DOCS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.toast(f"Saved: {uploaded_file.name}")

    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    pdf_names = [f.name for f in pdf_files]
    
    if "pdf_selector" not in st.session_state:
        st.session_state.pdf_selector = pdf_names[0] if pdf_names else None
        
    selected_pdf = st.selectbox(
        "选择阅读目标", pdf_names, 
        key="pdf_selector_ui",
        disabled=not config_ready
    )

# =============================================================================
# 4. 主界面逻辑
# =============================================================================
st.title(f"🚀 AI 科研辅助终端")

if not config_ready:
    st.info("👈 请在左侧侧边栏填入 API Key 以激活系统。")
    st.stop()

# --- 初始化/更新 Agent ---
current_agent_config = {
    "key": user_api_key,
    "url": user_base_url,
    "model": user_model_name,
    "user": st.session_state.user_session_id
}

if "agent" not in st.session_state or st.session_state.get("last_agent_config") != current_agent_config:
    with st.spinner("正在初始化 Agent..."):
        try:
            st.session_state.agent = ResearchAgent(
                session_id=st.session_state.user_session_id,
                api_key=user_api_key,
                base_url=user_base_url,
                model=user_model_name
            )
            st.session_state.last_agent_config = current_agent_config
            st.toast("Agent 已在线")
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 状态推断与自愈 ---
if "phase" not in st.session_state:
    file_phase = "init"
    if check_milestone("total.md"): file_phase = "done" 
    elif check_milestone(FILE_FINAL): file_phase = "final" 
    elif check_milestone(FILE_INNOV_3): file_phase = "final"
    elif check_milestone(FILE_INNOV_2): file_phase = "innov3"
    elif check_milestone(FILE_INNOV_1): file_phase = "innov2"
    elif check_milestone(FILE_BASE_INFO): file_phase = "innov1"
    st.session_state.phase = file_phase

state_changed = False
if st.session_state.phase == "read" and check_milestone(FILE_BASE_INFO):
    st.session_state.phase = "innov1"; state_changed = True
elif st.session_state.phase == "innov1" and check_milestone(FILE_INNOV_1):
    st.session_state.phase = "innov2"; state_changed = True
elif st.session_state.phase == "innov2" and check_milestone(FILE_INNOV_2):
    st.session_state.phase = "innov3"; state_changed = True
elif st.session_state.phase == "innov3" and check_milestone(FILE_INNOV_3):
    st.session_state.phase = "final"; state_changed = True

if state_changed: time.sleep(0.1); st.rerun()

# --- 渲染聊天历史 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 业务逻辑 Phase 分发 ---

# Phase: Init
if st.session_state.phase == "init":
    st.subheader("阶段一：文献输入")
    if not selected_pdf:
        st.info("请先在侧边栏上传或选择 PDF。")
    else:
        st.success(f"已选中: **{selected_pdf}**")
        if st.button("🚀 开始深度阅读", type="primary", use_container_width=True):
            st.session_state.phase = "read"
            if not (USER_RES_DIR / FILE_MEMORY).exists():
                with open(USER_RES_DIR / FILE_MEMORY, 'w', encoding='utf-8') as f:
                    f.write(PromptManager.get_memory_init_content())
            st.rerun()

# Phase: Read
elif st.session_state.phase == "read":
    if not check_milestone(FILE_BASE_INFO):
        if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
            trigger_msg = f"请读取文件 '{selected_pdf}'，深入分析并建立 '{FILE_BASE_INFO}'。"
            st.session_state.messages.append({"role": "user", "content": trigger_msg})
            st.rerun()
        
        if st.session_state.messages[-1]["role"] == "user":
            st.session_state.agent.update_phase("read")
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                full_response = ""
                try:
                    trigger_msg = st.session_state.messages[-1]["content"]
                    stream = st.session_state.agent.chat_stream(trigger_msg, callbacks=[st_callback])
                    res_slot = st.empty()
                    for chunk in stream:
                        full_response += chunk
                        res_slot.markdown(full_response + "▌")
                    res_slot.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun()
                except Exception as e:
                    st.error(f"执行错误: {e}")
    else:
        st.session_state.phase = "innov1"; st.rerun()

# Phase: Innovations
elif st.session_state.phase in ["innov1", "innov2", "innov3"]:
    phase_map = {"innov1": (FILE_INNOV_1, 1), "innov2": (FILE_INNOV_2, 2), "innov3": (FILE_INNOV_3, 3)}
    current_file, stage_num = phase_map[st.session_state.phase]

    if f"ready_{st.session_state.phase}" not in st.session_state:
        context = {"base_summary": read_file_content(FILE_BASE_INFO), "memory_log": read_file_content(FILE_MEMORY)}
        st.session_state.agent.update_phase(st.session_state.phase, context)
        st.session_state.agent.clear_short_term_memory()
        st.session_state.messages.append({"role": "assistant", "content": f"### 💡 创新点挖掘：第 {stage_num} 点\n\n系统就绪。请提出您的初步想法。"})
        st.session_state[f"ready_{st.session_state.phase}"] = True
        st.rerun()

    if prompt := st.chat_input(f"请输入关于创新点 {stage_num} 的想法..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            full_response = ""
            try:
                stream = st.session_state.agent.chat_stream(prompt, callbacks=[st_callback])
                res_slot = st.empty()
                for chunk in stream:
                    full_response += chunk
                    res_slot.markdown(full_response + "▌")
                res_slot.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                if check_milestone(current_file):
                    st.success("🎉 创新点已定稿！"); time.sleep(1); st.rerun()
            except Exception as e: st.error(f"Error: {e}")

# Phase: Final
elif st.session_state.phase == "final":
    st.subheader("🔬 最终实验设计")
    if not check_milestone(FILE_FINAL):
        if "final_triggered" not in st.session_state:
            context = {"base_summary": read_file_content(FILE_BASE_INFO)}
            st.session_state.agent.update_phase("final", context)
            st.session_state.agent.clear_short_term_memory()
            trigger = "所有创新点已配齐。请设计最终实验方案并写入 final_innov.md。"
            st.session_state.messages.append({"role": "user", "content": trigger})
            st.session_state["final_triggered"] = True
            st.rerun() 

        if st.session_state.messages[-1]["role"] == "user":
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                res_slot = st.empty()
                full_response = ""
                try:
                    trigger_text = st.session_state.messages[-1]["content"]
                    stream = st.session_state.agent.chat_stream(trigger_text, callbacks=[st_callback])
                    for chunk in stream:
                        full_response += chunk
                        res_slot.markdown(full_response + "▌")
                    res_slot.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    if check_milestone(FILE_FINAL): st.rerun() 
                except Exception as e:
                    st.error(f"Error: {str(e)}"); del st.session_state["final_triggered"]
    else:
        st.success("✅ 实验设计已完成。")
        if st.button("🏁 生成汇总报告", type="primary", use_container_width=True):
            merge_final_report(); st.session_state.phase = "done"; st.rerun()

# Phase: Done
elif st.session_state.phase == "done":
    st.header("🏆 提案完成")
    if not check_milestone("total.md"): merge_final_report()
    content = read_file_content("total.md")
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        if content: st.download_button("📥 下载 Markdown", content, f"proposal_{st.session_state.user_session_id}.md", type="primary", use_container_width=True)
    
    # 既然有重置按钮在侧边栏，这里可以简化，或者保留作为快捷入口
    st.success("任务已全部完成！如需开始新任务，请使用左侧边栏的【重置选项】。")
            
    st.divider()
    if content:
        with st.expander("📄 查看报告全文", expanded=True):
            st.markdown(content)