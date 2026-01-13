import sys
import time
import os
from pathlib import Path

# 引入我们的模块
from src.agent import ResearchAgent
from src.config import DOCS_DIR, RES_DIR, FILE_BASE_INFO, FILE_MEMORY, FILE_FINAL

# 定义颜色代码，让终端交互更清晰
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

def print_system(msg):
    print(f"{GREEN}[System]{RESET} {msg}")

def read_local_file(filename):
    """辅助函数：读取本地生成的文件内容，用于构建下一阶段的 Context"""
    path = RES_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def check_milestone_completed(filename):
    """检查是否已经生成了目标文件（标志着当前阶段结束）"""
    return (RES_DIR / filename).exists()

def main():
    # 0. 初始化
    print(f"{CYAN}=========================================================={RESET}")
    print(f"{CYAN}   AI Research Agent: Top-Tier Paper Innovation Assistant   {RESET}")
    print(f"{CYAN}=========================================================={RESET}")
    
    # 检查 docs 目录下是否有文件
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"{RED}Error: docs/ 目录下没有找到 PDF 文件。请放入论文后重试。{RESET}")
        return

    # 让用户选择文件（如果有多个）
    target_pdf = pdf_files[0].name
    if len(pdf_files) > 1:
        print(f"发现多个PDF文件:")
        for idx, f in enumerate(pdf_files):
            print(f"{idx + 1}. {f.name}")
        choice = input("请输入序号选择: ")
        try:
            target_pdf = pdf_files[int(choice)-1].name
        except:
            print("输入无效，默认选择第一个。")

    print_system(f"目标论文: {target_pdf}")
    print_system("初始化 Agent...")
    
    # 初始化记忆文件
    if not (RES_DIR / FILE_MEMORY).exists():
        from src.prompts import PromptManager
        with open(RES_DIR / FILE_MEMORY, 'w', encoding='utf-8') as f:
            f.write(PromptManager.get_memory_init_content())

    agent = ResearchAgent()

    # =========================================================================
    # Phase 1: Reading & Base Establishment
    # =========================================================================
    if not check_milestone_completed(FILE_BASE_INFO):
        print_system("进入阶段 1: 论文全量阅读与基准建立 (Base Extraction)")
        agent.update_phase("read")
        
        # 发送指令给 Agent
        trigger_msg = f"请读取文件 '{target_pdf}'，分析其核心方法和理论，并建立 '{FILE_BASE_INFO}'。"
        print(f"\n{YELLOW}>> {trigger_msg}{RESET}")
        
        response = agent.chat(trigger_msg)
        print(f"\n{CYAN}[Agent]:{RESET} {response}")
        
        # 强制检查：必须生成了 base.md 才能继续
        if not check_milestone_completed(FILE_BASE_INFO):
            print_system("警告：Agent 似乎没有成功创建 base.md。请手动检查或重试。")
            return
    else:
        print_system(f"检测到 {FILE_BASE_INFO} 已存在，跳过阅读阶段。")

    # =========================================================================
    # Phase 2: Innovation Loop (1 -> 2 -> 3)
    # =========================================================================
    # 定义三个创新点的文件名
    innov_files = ["innov1.md", "innov2.md", "innov3.md"]
    
    for i, innov_file in enumerate(innov_files):
        stage_num = i + 1
        
        # 如果这个创新点文件已经存在，就跳过（支持断点续传）
        if check_milestone_completed(innov_file):
            print_system(f"检测到 {innov_file} 已存在，跳过创新点 {stage_num}。")
            continue
            
        print_system(f"进入阶段 2-{stage_num}: 挖掘第 {stage_num} 个创新点")
        
        # 1. 准备上下文：读取 Base 和 之前的创新点
        context_data = {
            "base_summary": read_local_file(FILE_BASE_INFO),
            "memory_log": read_local_file(FILE_MEMORY),
            # 这里可以扩展：把之前的创新点内容也读进来
        }
        
        # 2. 切换大脑
        agent.update_phase(f"innov{stage_num}", context_data)
        agent.clear_short_term_memory() # 清空上一轮的对话缓存
        
        # 3. 交互循环
        print(f"\n{CYAN}>>> 请提出你关于创新点 {stage_num} 的初步思路 (输入 'q' 退出):{RESET}")
        
        first_turn = True
        while not check_milestone_completed(innov_file):
            if first_turn:
                user_input = input(f"{YELLOW}(Idea {stage_num}) You: {RESET}")
                first_turn = False
            else:
                user_input = input(f"{YELLOW}(Discussing {stage_num}) You: {RESET}")
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print_system("用户终止程序。")
                sys.exit(0)
            
            # Agent 执行
            response = agent.chat(user_input)
            print(f"\n{CYAN}[Agent]:{RESET} {response}")
            
            # 检查是否完成了任务
            if check_milestone_completed(innov_file):
                print(f"\n{GREEN}🎉 恭喜！创新点 {stage_num} 已定稿并归档。{RESET}")
                break
            else:
                print(f"\n{YELLOW}[提示] 创新点尚未归档。请继续与 Agent 讨论并确认方案，或输入 '确认该方案' 提示 Agent 写文件。{RESET}")

    # =========================================================================
    # Phase 3: Final Experiment Design
    # =========================================================================
    if not check_milestone_completed(FILE_FINAL):
        print_system("进入阶段 3: 最终实验方案设计 (Final Design)")
        
        # 准备所有素材
        context_data = {
            "base_summary": read_local_file(FILE_BASE_INFO),
            # 此时 innov1/2/3 肯定都存在了，agent tool 会自己去读，
            # 但我们在 Prompt 里并没有强制把全文塞进去，而是让 Agent 自己调用 read_file_tool
            # 这样可以节省 Context Token
        }
        
        agent.update_phase("final")
        agent.clear_short_term_memory()
        
        trigger_msg = "三个创新点已就绪。请读取所有 innov 文件，设计最终的对比实验和消融实验方案，并写入 final_innov.md。"
        print(f"\n{YELLOW}>> {trigger_msg}{RESET}")
        
        response = agent.chat(trigger_msg)
        print(f"\n{CYAN}[Agent]:{RESET} {response}")
    else:
        print_system(f"检测到 {FILE_FINAL} 已存在。项目似乎已完成。")

    print(f"\n{GREEN}=========================================================={RESET}")
    print(f"{GREEN}   Mission Complete! 所有文件已保存在 res/ 目录下。        {RESET}")
    print(f"{GREEN}=========================================================={RESET}")

if __name__ == "__main__":
    main()