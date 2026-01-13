import fitz  # PyMuPDF
import os
import logging

# 配置模块级日志
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    PDF 处理核心类。
    负责读取 PDF 文件并将其转换为带有页码标记的纯文本/Markdown 格式。
    """

    def __init__(self):
        pass

    def parse_pdf(self, file_path: str) -> str:
        """
        解析 PDF 文件内容。

        Args:
            file_path (str): PDF 文件的绝对或相对路径。

        Returns:
            str: 包含页码标记的完整文本内容。

        Raises:
            FileNotFoundError: 如果文件不存在。
            ValueError: 如果文件不是 PDF 格式。
            Exception: 其他解析错误。
        """
        # 1. 基础校验
        if not os.path.exists(file_path):
            error_msg = f"PDF file not found at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not file_path.lower().endswith('.pdf'):
            error_msg = f"The file '{file_path}' does not appear to be a PDF."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Starting to parse PDF: {file_path}")
        
        full_text = []
        
        try:
            # 2. 打开文档
            with fitz.open(file_path) as doc:
                total_pages = doc.page_count
                logger.info(f"Document has {total_pages} pages.")

                # 3. 逐页提取
                for page_num, page in enumerate(doc, start=1):
                    # 提取纯文本 (flags=0 保持最基础的读取，也可以尝试 "blocks" 做更复杂的布局分析)
                    # 这里选择 "text" 模式，因为它对 LLM 的 Token 消耗最友好且保留了阅读顺序
                    text = page.get_text("text")
                    
                    # 清洗文本：去除多余的首尾空白
                    clean_text = text.strip()

                    # 4. 注入页码锚点 (关键步骤)
                    # 格式设计为明显的分隔符，方便 LLM 识别
                    header = f"\n\n=== Page {page_num} ===\n\n"
                    
                    if clean_text:
                        full_text.append(header + clean_text)
                    else:
                        # 即使是空页(如图片页)，保留页码标记也是好的，防止幻觉
                        full_text.append(header + "[Content is empty or image-only]")

            logger.info("PDF parsing completed successfully.")
            return "".join(full_text)

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise e

    def save_markdown(self, content: str, output_path: str) -> None:
        """
        将处理后的文本保存为 Markdown 文件。

        Args:
            content (str): 要保存的文本内容。
            output_path (str): 输出文件路径。
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Content saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save markdown file: {e}")
            raise e