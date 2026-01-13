import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# 引入核心 Agent
from src.analysis.reviewer import ReviewAgent

# 加载本地环境作为备用（仅用于开发调试，生产环境由用户输入 Key）
load_dotenv()

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="AutoPaperReviewer - 多用户并发版",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 美化
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 20px;}
    .report-box {background-color: #f9f9f9; padding: 25px; border-radius: 10px; border: 1px solid #ddd; line-height: 1.6;}
    .user-instruction {
        background-color: #e3f2fd; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #1E88E5; 
        margin-bottom: 20px;
        font-size: 0.9em;
    }
    .stButton>button {width: 100%; border-radius: 8px; height: 50px; font-weight: bold;}
    .safe-badge {
        background-color: #d4edda; color: #155724; padding: 5px 10px; 
        border-radius: 15px; font-size: 0.8em; font-weight: bold; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 侧边栏：会话级配置 (Session Scoped)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ 个人设置")
    st.markdown("<div class='safe-badge'>🔒 会话隔离模式</div>", unsafe_allow_html=True)
    st.markdown("配置仅在当前浏览器会话有效，互不干扰。")
    
    st.markdown("---")
    
    # --- 1. 服务商选择 ---
    base_url_options = {
        "OpenAI (官方)": "https://api.openai.com/v1",
        "DeepSeek (深度求索)": "https://api.deepseek.com/v1",
        "Moonshot (Kimi)": "https://api.moonshot.cn/v1",
        "自定义 (Custom)": "custom"
    }
    
    selected_provider = st.selectbox("1. 服务商 / Base URL", list(base_url_options.keys()))
    
    if selected_provider == "自定义 (Custom)":
        user_base_url = st.text_input("请输入 Base URL", value="https://api.openai.com/v1")
    else:
        user_base_url = st.text_input("Base URL", value=base_url_options[selected_provider], disabled=False)

    # --- 2. API Key ---
    user_api_key = st.text_input(
        "2. API Key (sk-...)", 
        type="password", 
        help="您的 Key 将直接传给 Agent 实例，不经过全局环境变量。",
        placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    )

    # --- 3. 模型名称 ---
    default_models = {
        "OpenAI (官方)": "gpt-4o",
        "DeepSeek (深度求索)": "deepseek-chat",
        "Moonshot (Kimi)": "moonshot-v1-32k",
        "自定义 (Custom)": "gpt-4o"
    }
    
    user_model_name = st.text_input(
        "3. 模型名称 (Model Name)", 
        value=default_models.get(selected_provider, "gpt-4o"),
        help="例如: gpt-4o, gpt-4-turbo, deepseek-chat"
    )

    # --- 4. 温度参数 ---
    st.markdown("---")
    temp_value = st.slider("Temperature (创造性)", 0.0, 1.0, 0.2, 0.1)

    # 状态检查
    config_ready = bool(user_api_key and user_base_url and user_model_name)
    if config_ready:
        st.success("✅ 配置已就绪")
    else:
        st.warning("⚠️ 请完整填写上方信息")


# ==========================================
# 3. 主界面逻辑
# ==========================================
st.markdown('<div class="main-header">📑 科研论文自动审稿 Agent</div>', unsafe_allow_html=True)

# 阻断逻辑
if not config_ready:
    st.info("👈 请先在左侧侧边栏配置您的 API Key 以开始独立的审稿会话。")
    st.stop()

# --- 文件上传 ---
uploaded_file = st.file_uploader("📂 上传论文 PDF (Drag and drop)", type=["pdf"])

if uploaded_file is not None:
    # --- 交互式意图对齐区域 ---
    st.markdown("### 🎯 审稿意图对齐 (Customize Your Review)")
    st.markdown("告诉 Agent 您希望它**重点关注**什么。")

    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            focus_tags = st.multiselect(
                "快速选择关注点:",
                ["数学推导严谨性", "实验数据可信度", "创新性评估", "相关工作遗漏", "逻辑自洽性", "代码/复现可行性", "投稿建议"],
                help="Agent 会在审稿报告中优先分析这些维度。"
            )
        with col2:
            custom_text_input = st.text_area(
                "补充自然语言指令:",
                placeholder="例如：请特别检查第3章的公式推导是否正确...",
                height=100
            )

    # --- 执行按钮 ---
    if st.button("🚀 开始深度审稿 (Start Review)", type="primary"):
        
        # 1. 构造用户指令
        combined_instructions = []
        if focus_tags:
            combined_instructions.append(f"重点关注维度: {', '.join(focus_tags)}")
        if custom_text_input:
            combined_instructions.append(f"用户具体指令: {custom_text_input}")
        
        final_instruction_str = "\n".join(combined_instructions)

        # 2. 生成唯一文件路径 (防止多用户文件冲突)
        unique_id = str(uuid.uuid4())[:8]  # 生成短 UUID
        safe_filename = f"{unique_id}_{uploaded_file.name}"
        
        input_dir = "data/input"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            
        temp_pdf_path = os.path.join(input_dir, safe_filename)
        
        # 保存文件
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 3. Agent 执行流程 (隔离模式)
        try:
            if final_instruction_str:
                st.markdown(f'<div class="user-instruction"><b>已注入自定义指令:</b><br>{final_instruction_str.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            with st.status("正在进行深度分析 (Deep Analysis)...", expanded=True) as status:
                
                st.write(f"🔌 初始化独立 Agent 实例 (模型: {user_model_name})...")
                
                # [关键] 实例化 Agent，直接传入 Key，不使用环境变量
                agent = ReviewAgent(
                    api_key=user_api_key, 
                    base_url=user_base_url, 
                    model=user_model_name
                )
                
                # 应用当前会话的温度设置
                agent.config['llm']['temperature'] = temp_value
                
                st.write("📖 解析 PDF 文档结构...")
                
                # 调用核心 Review 方法
                review_content = agent.review(temp_pdf_path, custom_instructions=final_instruction_str)
                
                st.write("💾 保存报告到本地...")
                agent.save_report(review_content, temp_pdf_path)
                
                status.update(label="✅ 审稿完成!", state="complete", expanded=False)

            # 4. 结果展示
            st.success("分析完成！")
            
            tab1, tab2 = st.tabs(["📝 审稿报告预览", "🔍 原始 Markdown"])
            with tab1:
                st.markdown(f'<div class="report-box">{review_content}</div>', unsafe_allow_html=True)
            with tab2:
                st.code(review_content, language="markdown")

            # 5. 下载按钮
            st.download_button(
                label="📥 下载报告 (.md)",
                data=review_content,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_Review.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"❌ 发生错误: {str(e)}")
            st.error("请检查 API Key 是否正确，或网络是否通畅。")
            with st.expander("调试信息"):
                st.write(e)
else:
    st.info("👋 请上传 PDF 文件以激活审稿面板。")