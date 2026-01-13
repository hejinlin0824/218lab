import os
import fitz  # PyMuPDF
from pathlib import Path
from langchain.tools import StructuredTool
from src.config import DOCS_DIR
from langchain_community.tools.tavily_search import TavilySearchResults

class ToolFactory:
    """
    工具工厂：为每个会话动态生成绑定了特定目录的工具集。
    实现多用户环境下的文件读写隔离。
    """
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.figures_dir = self.session_dir / "figures"
        
        # 确保当前会话的图片存储目录存在
        if not self.figures_dir.exists():
            self.figures_dir.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, file_name: str) -> Path:
        """
        安全检查：确保文件路径限定在当前用户的会话目录内。
        防止路径穿越攻击 (e.g., ../../other_user/file)。
        """
        target_path = (self.session_dir / file_name).resolve()
        base_path = self.session_dir.resolve()
        try:
            common = os.path.commonpath([str(base_path), str(target_path)])
        except ValueError:
            raise ValueError(f"Path jump detected. Access denied for {file_name}")

        if str(common) != str(base_path):
             raise ValueError(f"Path traversal attempt. Access to {file_name} is denied.")
        return target_path

    def get_tools(self):
        """
        返回绑定了当前 session_dir 的 LangChain 工具列表。
        """

        # --- 1. 定义具体的工具函数逻辑 (闭包) ---
        
        def read_paper_func(pdf_filename: str) -> str:
            """
            读取论文 PDF 内容。
            逻辑：
            1. 从全局 docs 目录读取原始 PDF。
            2. 提取文本。
            3. 将提取的图片保存到用户的 session/figures 目录，实现资源隔离。
            """
            # 默认去全局 DOCS_DIR 找文件
            pdf_path = DOCS_DIR / pdf_filename
            
            if not pdf_path.exists():
                return f"Error: 文件 {pdf_filename} 在 docs 目录下未找到。"

            try:
                full_text = []
                image_counter = 0
                pdf_name_stem = pdf_path.stem

                with fitz.open(pdf_path) as doc:
                    total_pages = len(doc)
                    for page_index, page in enumerate(doc):
                        page_image_notes = []
                        
                        # --- 图片提取逻辑 ---
                        try:
                            image_list = page.get_images(full=True)
                            if image_list:
                                for img_index, img in enumerate(image_list):
                                    try:
                                        xref = img[0]
                                        base_image = doc.extract_image(xref)
                                        image_bytes = base_image["image"]
                                        image_ext = base_image["ext"]
                                        
                                        # 构造文件名并保存到【当前用户的】figures 目录
                                        image_filename = f"{pdf_name_stem}_p{page_index + 1}_img{img_index + 1}.{image_ext}"
                                        image_save_path = self.figures_dir / image_filename
                                        
                                        with open(image_save_path, "wb") as f:
                                            f.write(image_bytes)
                                        
                                        image_counter += 1
                                        # 返回相对路径，方便 Markdown 引用
                                        rel_path = os.path.relpath(image_save_path, self.session_dir)
                                        page_image_notes.append(f"\n[Image Reference: Figure saved at {rel_path}]")
                                    except Exception: 
                                        continue
                        except Exception:
                            pass
                        
                        # --- 文本提取逻辑 ---
                        text = page.get_text("text")
                        clean_text = text.strip()
                        page_content = f"\n--- Page {page_index + 1} ---\n{clean_text}\n" + "\n".join(page_image_notes)
                        full_text.append(page_content)

                summary_info = f"[System Note: Successfully read {total_pages} pages. Extracted {image_counter} images to {self.figures_dir}.]\n\n"
                return summary_info + "\n".join(full_text)
            except Exception as e:
                return f"Critical Error processing PDF '{pdf_filename}': {str(e)}"

        def write_file_func(file_name: str, content: str) -> str:
            """
            写入 Markdown 文件到当前用户的会话目录。
            """
            try:
                file_path = self._validate_path(file_name)
                # 确保父目录存在
                if not file_path.parent.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                rel_path = os.path.relpath(file_path, self.session_dir)
                return f"Successfully wrote content to {rel_path}"
            except Exception as e:
                return f"Error writing file: {str(e)}"

        def read_file_func(file_name: str) -> str:
            """
            从当前用户的会话目录读取文件。
            """
            try:
                file_path = self._validate_path(file_name)
                if not file_path.exists():
                    return f"Error: File {file_name} does not exist."
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"

        def web_search_func(query: str) -> str:
            """
            网络搜索工具 (Tavily)。
            """
            api_key = os.getenv("T_SEARCH_API")
            if not api_key: 
                return "System Error: 'T_SEARCH_API' not found in environment variables."
            
            try:
                # 限制 max_results 以节省 token
                tool = TavilySearchResults(tavily_api_key=api_key, max_results=5)
                results = tool.invoke({"query": query})
                
                if not results:
                    return f"No results found for query: {query}"
                
                formatted_output = [f"Search Results for '{query}':\n"]
                for idx, res in enumerate(results, 1):
                    content = res.get('content', 'No content')
                    url = res.get('url', 'No URL')
                    formatted_output.append(f"Source {idx}: {content}\n(Link: {url})\n")
                
                return "\n".join(formatted_output)
            except Exception as e:
                return f"Search execution failed: {str(e)}"

        # --- 2. 包装并返回 StructuredTool 列表 ---
        return [
            StructuredTool.from_function(
                func=read_paper_func,
                name="read_paper_tool",
                description="Useful for reading the content of a research paper PDF file. Input should be the filename of the pdf (e.g., 'paper.pdf') located in the docs directory."
            ),
            StructuredTool.from_function(
                func=write_file_func,
                name="write_file_tool",
                description="Useful for writing content to a markdown file in the resource directory. Input should be the file_name (e.g., 'base.md', 'innov1.md') and the full content string."
            ),
            StructuredTool.from_function(
                func=read_file_func,
                name="read_file_tool",
                description="Useful for reading existing markdown files from the resource directory."
            ),
            StructuredTool.from_function(
                func=web_search_func,
                name="web_search_tool",
                description="Useful for searching the internet to check if an idea already exists (Novelty Check) or to find theoretical references."
            )
        ]