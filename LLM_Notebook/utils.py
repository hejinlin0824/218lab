import os
from pathlib import Path

def list_markdown_files(user_root: Path, exclude_list: list) -> list:
    """
    递归扫描研究员目录下所有的 Markdown 文件，用于侧边栏的“查”功能。
    
    :param user_root: 研究员的专属根目录 (例如 res/hejinlin)
    :param exclude_list: 排除列表，如 ['faiss_index', 'figures', 'memory.md']
    :return: 相对路径列表，按字母顺序排序
    """
    md_files = []
    
    if not user_root.exists():
        return []

    for root, dirs, files in os.walk(user_root):
        # 原地修改 dirs 以排除不需要显示的系统文件夹
        dirs[:] = [d for d in dirs if d not in exclude_list and not d.startswith('.')]
        
        for file in files:
            # 仅列出 .md 文件，且过滤掉 exclude_list 中的特定文件名
            if file.endswith(".md") and file not in exclude_list:
                full_path = Path(root) / file
                # 计算相对于用户根目录的相对路径，方便 UI 显示
                rel_path = full_path.relative_to(user_root)
                md_files.append(str(rel_path))
                
    return sorted(md_files)

def read_file_content(file_path: Path) -> str:
    """
    读取笔记内容。
    
    :param file_path: 文件的绝对路径
    :return: 文件文本内容，不存在则返回空字符串
    """
    if not file_path.exists():
        return ""
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def save_file_content(file_path: Path, content: str) -> tuple:
    """
    保存/更新笔记内容。
    
    :param file_path: 目标文件路径
    :param content: Markdown 格式的文本内容
    :return: (bool, message) 成功状态及反馈
    """
    try:
        # 自动创建不存在的父级目录（支持在子目录下新建笔记）
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True, "✅ 笔记保存成功"
    except Exception as e:
        return False, f"❌ 保存失败: {str(e)}"

def delete_file(file_path: Path) -> tuple:
    """
    物理删除文件。
    """
    try:
        if file_path.exists():
            os.remove(file_path)
            return True, "🗑️ 文件已永久删除"
        return False, "❌ 错误：文件不存在"
    except Exception as e:
        return False, f"❌ 删除错误: {str(e)}"

def validate_path_security(target_path: Path, user_root: Path) -> bool:
    """
    路径安全检查：防止研究员通过 ../ 路径删除或查看他人的文件。
    """
    try:
        abs_target = target_path.resolve()
        abs_root = user_root.resolve()
        # 确保目标路径的前缀必须是该用户的根目录
        return str(abs_target).startswith(str(abs_root))
    except Exception:
        return False