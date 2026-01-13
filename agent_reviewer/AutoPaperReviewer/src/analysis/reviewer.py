import yaml
import os
import logging
from typing import Dict, Any, Optional

from src.ingestion.pdf_processor import PDFProcessor
from src.core.llm_client import LLMClient

# 配置模块级日志
logger = logging.getLogger(__name__)

class ReviewAgent:
    """
    审稿 Agent 核心类。
    负责编排整个审稿流程。
    
    并发支持说明：
    实例化时接收特定的 api_key/base_url，从而创建独立的 LLMClient 实例，
    确保多用户环境下的会话隔离。
    """

    def __init__(self, 
                 api_key: str = None, 
                 base_url: str = None, 
                 model: str = None,
                 config_path: str = "config/settings.yaml", 
                 prompt_path: str = "config/prompts.yaml"):
        """
        初始化 ReviewAgent。

        Args:
            api_key (str): 用户的 API Key (用于多用户隔离)。
            base_url (str): 用户的 API Base URL。
            model (str): 用户选择的模型名称。
            config_path (str): 全局配置文件路径。
            prompt_path (str): 提示词文件路径。
        """
        self.config = self._load_yaml(config_path)
        self.prompts = self._load_yaml(prompt_path)
        
        # 如果初始化时传入了特定模型，覆盖配置文件中的默认值
        if model:
            logger.info(f"Overriding default model with: {model}")
            self.config['llm']['default_model'] = model

        self.pdf_processor = PDFProcessor()
        
        # 关键：将凭证传递给 LLMClient，而不是依赖全局环境变量
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url, model=model)
        
        logger.info("ReviewAgent initialized with session-specific credentials.")

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def review(self, pdf_path: str, custom_instructions: str = "") -> str:
        """
        执行审稿任务。

        Args:
            pdf_path (str): 待审阅 PDF 的路径。
            custom_instructions (str): 用户自定义的审稿指令（意图对齐）。

        Returns:
            str: 生成的审稿报告内容。
        """
        logger.info(f"Starting review for: {pdf_path}")

        # 1. 解析 PDF 获取带页码的文本
        paper_content = self.pdf_processor.parse_pdf(pdf_path)
        
        content_len = len(paper_content)
        logger.info(f"Parsed content length: {content_len} characters")
        if content_len < 100:
            logger.warning("Paper content seems too short. Is this a valid PDF?")

        # 2. 构建 Prompt
        system_prompt = self.prompts.get('reviewer_system', '')
        if not system_prompt:
            raise ValueError("Key 'reviewer_system' not found in prompts.yaml")

        user_template = self.prompts.get('user_template', '')
        if not user_template:
            raise ValueError("Key 'user_template' not found in prompts.yaml")
        
        # --- 注入用户意图 (Intent Alignment) ---
        user_intent_block = ""
        if custom_instructions:
            user_intent_block = f"""
\n\n
!!! USER SPECIAL INSTRUCTIONS (HIGHEST PRIORITY) !!!
The user has specified the following focus areas. Please adjust your review to prioritize these points:
{custom_instructions}
!!! END INSTRUCTIONS !!!
\n\n
"""
        
        # 组装最终 User Prompt
        user_prompt = user_template.format(paper_content=paper_content) + user_intent_block

        # 3. 调用 LLM
        temperature = self.config['llm'].get('temperature', 0.2)
        model = self.config['llm'].get('default_model', None)

        logger.info("Sending prompt to LLM...")
        review_report = self.llm_client.get_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature
        )

        logger.info("Review generated successfully.")
        return review_report

    def save_report(self, content: str, original_pdf_path: str) -> str:
        """
        保存审稿报告到指定目录。
        """
        output_dir = self.config['paths']['output_dir']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 基于传入的 original_pdf_path 生成文件名
        # 注意：GUI 层已经处理了 UUID 唯一化，这里直接使用文件名即可
        base_name = os.path.basename(original_pdf_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_filename = f"{file_name_no_ext}_Review.md"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Report saved to: {output_path}")
        return output_path