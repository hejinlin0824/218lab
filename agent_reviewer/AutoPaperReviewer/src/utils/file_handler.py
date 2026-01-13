import os
import re
import shutil

class FileHandler:
    """
    文件操作通用工具类。
    """
    
    @staticmethod
    def ensure_directory(path: str):
        """确保目录存在，不存在则创建"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def clean_filename(filename: str) -> str:
        """
        清理文件名，移除非法字符，防止保存文件时出错。
        例如: "Paper: AI & Future?" -> "Paper_AI_and_Future"
        """
        # 替换常见非法字符
        filename = filename.replace(':', '_').replace('/', '_').replace('\\', '_')
        # 移除非 ASCII 符号或控制字符 (可选)
        filename = re.sub(r'[^\w\-. ]', '', filename)
        return filename.strip()

    @staticmethod
    def verify_extension(file_path: str, allowed_extensions: list) -> bool:
        """
        验证文件后缀是否合法。
        Example: verify_extension("test.pdf", [".pdf"]) -> True
        """
        _, ext = os.path.splitext(file_path)
        return ext.lower() in [e.lower() for e in allowed_extensions]

    @staticmethod
    def read_text(file_path: str) -> str:
        """读取文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def write_text(file_path: str, content: str):
        """写入文本文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)