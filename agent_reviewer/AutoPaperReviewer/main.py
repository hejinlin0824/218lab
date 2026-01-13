import os
import sys
import argparse
import logging
from src.analysis.reviewer import ReviewAgent

def setup_logging(verbose: bool = True):
    """
    配置全局日志格式。
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(
        description="AutoPaperReviewer: 基于 LLM 的科研论文自动审稿 Agent"
    )
    
    parser.add_argument(
        "pdf_path", 
        type=str, 
        help="待审阅论文 PDF 文件的路径 (e.g., data/input/paper.pdf)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="指定使用的 LLM 模型 (覆盖配置文件)"
    )

    args = parser.parse_args()

    # 2. 初始化日志
    setup_logging()
    logger = logging.getLogger("Main")

    # 3. 校验输入文件
    if not os.path.exists(args.pdf_path):
        logger.error(f"Input file not found: {args.pdf_path}")
        sys.exit(1)

    try:
        logger.info("Initializing Review Agent...")
        
        # 4. 初始化 Agent
        agent = ReviewAgent()
        
        # 如果命令行指定了模型，临时修改配置 (可选功能)
        if args.model:
            logger.info(f"Overriding model configuration to: {args.model}")
            agent.config['llm']['default_model'] = args.model

        # 5. 执行核心任务
        print("\n" + "="*60)
        print(f"🚀  Starting Review for: {os.path.basename(args.pdf_path)}")
        print("="*60 + "\n")

        review_report = agent.review(args.pdf_path)

        # 6. 保存结果
        saved_path = agent.save_report(review_report, args.pdf_path)

        # 7. 完成反馈
        print("\n" + "="*60)
        print("✅  Review Completed Successfully!")
        print(f"📄  Report saved to: {saved_path}")
        print("="*60 + "\n")

    except Exception as e:
        logger.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()