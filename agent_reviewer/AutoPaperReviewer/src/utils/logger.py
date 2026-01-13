import logging
import sys
import os
from pathlib import Path

def setup_logger(name: str, log_dir: str = "logs", verbose: bool = True) -> logging.Logger:
    """
    配置并返回一个标准的 Logger 实例。
    
    Args:
        name (str): Logger 的名称 (通常传入 __name__)。
        log_dir (str): 日志文件存放目录。
        verbose (bool): 是否在控制台输出 INFO 级别日志。
        
    Returns:
        logging.Logger: 配置好的 Logger 对象。
    """
    # 创建 Logger
    logger = logging.getLogger(name)
    
    # 如果已经有 Handler，避免重复添加 (防止日志重复打印)
    if logger.hasHandlers():
        return logger

    # 设置默认级别
    logger.setLevel(logging.DEBUG)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.INFO if verbose else logging.WARNING
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 Handler (RotatingFileHandler 也可以，这里用简单的 FileHandler)
    # 确保日志目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = os.path.join(log_dir, "app.log")
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # 文件中记录所有细节
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger