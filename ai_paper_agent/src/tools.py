import os
import fitz  # PyMuPDF
from typing import Optional, List
from langchain.tools import tool
from pathlib import Path
from src.config import DOCS_DIR, RES_DIR

# --- 新增依赖：Tavily Search ---
# 确保已运行: pip install tavily-python
from langchain_community.tools.tavily_search import TavilySearchResults

# 确保图片存储目录存在
FIGURES_DIR = RES_DIR / "figures"
if not FIGURES_DIR.exists():
    FIGURES_DIR.mkdir(parents=True)

class SecurityUtils:
    @staticmethod
    def validate_path(file_name: str, base_dir: Path) -> Path:
        target_path = (base_dir / file_name).resolve()
        base_path = base_dir.resolve()
        try:
            common = os.path.commonpath([str(base_path), str(target_path)])
        except ValueError:
            raise ValueError(f"Path jump detected. Access denied for {file_name}")

        if str(common) != str(base_path):
             raise ValueError(f"Path traversal attempt. Access to {file_name} is denied.")
        return target_path

class PaperReader:
    @staticmethod
    def extract_content(pdf_filename: str) -> str:
        """
        核心逻辑：读取PDF，提取文本，并将图片保存到本地作为证据。
        采用防御性编程，防止 PyMuPDF 'document closed' 错误。
        """
        pdf_path = None
        try:
            # 1. 路径验证
            try:
                pdf_path = SecurityUtils.validate_path(pdf_filename, DOCS_DIR)
            except Exception as e:
                return f"Error: 文件路径非法. {str(e)}"
            
            if not pdf_path.exists():
                return f"Error: 文件 {pdf_filename} 在 {DOCS_DIR} 目录下未找到。"
            
            # 2. 使用上下文管理器打开 PDF
            full_text = []
            image_counter = 0
            pdf_name_stem = pdf_path.stem

            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                # 3. 逐页解析
                for page_index in range(total_pages):
                    try:
                        page = doc.load_page(page_index)
                        
                        # --- 3.1 提取图片 ---
                        page_image_notes = []
                        image_list = page.get_images(full=True)
                        
                        if image_list:
                            for img_index, img in enumerate(image_list):
                                try:
                                    xref = img[0]
                                    base_image = doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    image_ext = base_image["ext"]
                                    
                                    image_filename = f"{pdf_name_stem}_p{page_index + 1}_img{img_index + 1}.{image_ext}"
                                    image_save_path = FIGURES_DIR / image_filename
                                    
                                    with open(image_save_path, "wb") as f:
                                        f.write(image_bytes)
                                    
                                    image_counter += 1
                                    rel_path = os.path.relpath(image_save_path, RES_DIR)
                                    page_image_notes.append(f"\n[Image Reference: Figure saved at res/{rel_path}]")
                                except Exception:
                                    # 图片提取失败不应阻断文本读取，跳过即可
                                    continue

                        # --- 3.2 提取文本 ---
                        text = page.get_text("text")
                        
                        page_content = f"\n--- Page {page_index + 1} ---\n{text}\n" + "\n".join(page_image_notes)
                        full_text.append(page_content)
                        
                    except Exception as page_err:
                        full_text.append(f"\n[System Error reading Page {page_index + 1}: {str(page_err)}]\n")

                summary_info = f"[System Note: Successfully read {total_pages} pages. Extracted {image_counter} images to {FIGURES_DIR}.]\n\n"
                return summary_info + "\n".join(full_text)

        except Exception as e:
            return f"Critical Error processing PDF '{pdf_filename}': {str(e)}"

# --- LangChain Tools 定义 ---

@tool
def read_paper_tool(pdf_filename: str) -> str:
    """
    Useful for reading the content of a research paper PDF file. 
    Input should be the filename of the pdf (e.g., 'paper.pdf') located in the docs directory.
    """
    return PaperReader.extract_content(pdf_filename)

@tool
def write_file_tool(file_name: str, content: str) -> str:
    """
    Useful for writing content to a markdown file in the resource directory.
    Input should be the file_name (e.g., 'base.md', 'innov1.md') and the full content string.
    """
    try:
        file_path = SecurityUtils.validate_path(file_name, RES_DIR)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        rel_path = os.path.relpath(file_path, RES_DIR)
        return f"Successfully wrote content to {rel_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool
def read_file_tool(file_name: str) -> str:
    """
    Useful for reading existing markdown files from the resource directory.
    """
    try:
        file_path = SecurityUtils.validate_path(file_name, RES_DIR)
        if not file_path.exists():
            return f"Error: File {file_name} does not exist."
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def web_search_tool(query: str) -> str:
    """
    Useful for searching the internet to check if an idea already exists (Novelty Check) or to find theoretical references.
    Input should be a specific search query string (e.g., "Federated Learning KL-divergence regularization survey").
    """
    # 1. 获取 API Key
    api_key = os.getenv("T_SEARCH_API")
    if not api_key:
        return "System Error: 'T_SEARCH_API' not found in environment variables. Please check .env file."

    try:
        # 2. 初始化 Tavily 工具
        # max_results=5 既能获取足够信息，又能控制 token 消耗
        tool = TavilySearchResults(tavily_api_key=api_key, max_results=5)
        
        # 3. 执行搜索
        results = tool.invoke({"query": query})
        
        # 4. 格式化输出 (将 List[Dict] 转为易读的 String)
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

# 注册所有工具
ALL_TOOLS = [read_paper_tool, write_file_tool, read_file_tool, web_search_tool]