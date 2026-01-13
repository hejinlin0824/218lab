# AutoPaperReviewer (科研论文自动审稿 Agent)

这是一个基于 LLM (Large Language Model) 的自动化科研论文审稿系统。它能够读取 PDF 格式的论文，基于深度学习与计算机科学领域的审稿标准，生成包含总结、优缺点分析、批判性疑问及评分的完整审稿报告。

## ✨ 核心功能

1.  **PDF 精准解析**：利用 `pymupdf` 提取论文内容，并自动保留页码信息 (e.g., `[Page 3]`)，实现有据可依的审稿。
2.  **专业审稿 Prompt**：基于 CO-STAR 框架设计的 Prompt，模拟顶级会议 (CVPR/NeurIPS) 领域主席的审稿视角。
3.  **多模型支持**：支持 OpenAI GPT-4, DeepSeek (深度求索), Moonshot (Kimi) 等兼容 OpenAI 接口的模型。
4.  **结构化输出**：自动生成 Markdown 格式的审稿报告，包含摘要、优缺点、评分等模块。

## 📂 项目结构

```text
AutoPaperReviewer/
├── config/                 # 配置文件 (Prompt, API设置)
├── data/                   # 数据目录
│   ├── input/              # 放入待审稿的 PDF
│   └── output/             # 生成的 Markdown 报告
├── src/                    # 源代码
│   ├── analysis/           # 审稿逻辑与标准
│   ├── core/               # LLM API 客户端
│   ├── ingestion/          # PDF 解析模块
│   └── utils/              # 通用工具 (日志, 文件操作)
└── main.py                 # 启动入口

🚀 快速开始
1. 环境准备
确保安装 Python 3.8+。

Bash

# 安装依赖
pip install -r requirements.txt
2. 配置 API Key
复制 .env 文件并填入你的 API Key：

Ini, TOML

# 编辑 .env 文件
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
# 如果使用 DeepSeek:
# OPENAI_BASE_URL=[https://api.deepseek.com/v1](https://api.deepseek.com/v1)
# LLM_MODEL=deepseek-chat
3. 运行审稿
将你的论文 PDF (例如 paper.pdf) 放入 data/input/ 目录，然后运行：

Bash

python main.py data/input/paper.pdf
4. 查看结果
程序运行完成后，请前往 data/output/ 目录查看生成的 paper_Review.md。

🛠️ 高级配置
修改审稿风格：编辑 config/prompts.yaml，调整 System Prompt 中的角色设定或评分标准。

调整模型参数：编辑 config/settings.yaml，修改 temperature 或 max_tokens。

📝 License
MIT License


---

### 文件 13: 必要的 `__init__.py` 文件

Python 包结构需要 `__init__.py` 文件。虽然内容为空，但必须创建以保证模块导入路径正确。

请在终端执行以下命令一次性补全所有缺失的 `__init__.py`：

```bash
touch src/__init__.py \
      src/ingestion/__init__.py \
      src/core/__init__.py \
      src/analysis/__init__.py \
      src/utils/__init__.py
或者你可以手动在上述每个文件夹下新建一个名为 __init__.py 的空文件。