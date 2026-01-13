import sys
from typing import Literal
from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

# 引入历史记录管理
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import llm, RES_DIR
# [修改点 1] 引入 ToolFactory 而不是静态的 ALL_TOOLS
from src.tools import ToolFactory
from src.prompts import PromptManager

class ResearchAgent:
    """
    科研Agent核心类。
    支持多会话并发，每个实例绑定一个 session_id。
    """
    
    def __init__(self, session_id: str):
        """
        初始化 Agent 实例。
        :param session_id: 用户的唯一会话 ID (UUID string)
        """
        self.session_id = session_id
        
        # [修改点 2] 动态构建用户专属目录
        self.session_dir = RES_DIR / self.session_id
        if not self.session_dir.exists():
            self.session_dir.mkdir(parents=True, exist_ok=True)
            
        self.llm = llm
        
        # [修改点 3] 使用工厂生成绑定了特定路径的工具集
        # 这样 Agent 调用的 write_file_tool 就会自动写到 res/{session_id}/ 下
        self.tools = ToolFactory(self.session_dir).get_tools()
        
        self.chat_history = ChatMessageHistory()
        self.agent_executor = None
        
    def _build_agent(self, system_prompt_content: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_content),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, # 保持 True 以便在终端看到思考过程
            handle_parsing_errors=True
        )

        self.agent_executor = RunnableWithMessageHistory(
            executor,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def update_phase(self, phase: Literal["read", "innov1", "innov2", "innov3", "final"], context_data: dict = None):
        # 打印日志带上 User ID 前缀，方便后台区分
        user_prefix = f"[User {self.session_id[:8]}]"
        print(f"\n[System] {user_prefix} Switching Agent Brain to Phase: {phase.upper()}...")
        
        # --- 递归上下文注入 ---
        dependencies = {
            "innov2": ["innov1.md"],
            "innov3": ["innov1.md", "innov2.md"],
            "final":  ["innov1.md", "innov2.md", "innov3.md"]
        }
        
        accumulated_context = ""
        if phase in dependencies:
            print(f"[System] {user_prefix} Loading previous context for coherence check...")
            for filename in dependencies[phase]:
                # [修改点 4] 从用户的 session_dir 读取前序文件，防止串号
                file_path = self.session_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            accumulated_context += f"\n\n=== [Context: {filename}] (Already Established) ===\n{content}\n"
                    except Exception as e:
                        print(f"[Warning] Failed to load context file {filename}: {e}")
        
        if context_data is None:
            context_data = {}
        context_data["prev_innovations"] = accumulated_context

        prompt_content = ""
        if phase == "read":
            prompt_content = PromptManager.get_phase1_prompt()
        elif phase in ["innov1", "innov2", "innov3"]:
            stage_num = int(phase[-1])
            prompt_content = PromptManager.get_innovation_prompt(stage_num, context_data or {})
        elif phase == "final":
            prompt_content = PromptManager.get_final_prompt()
        else:
            raise ValueError(f"Unknown phase: {phase}")

        self._build_agent(prompt_content)
        print(f"[System] {user_prefix} Agent is ready with new instructions (Context Injected).")

    def chat_stream(self, user_input: str, callbacks: list = None):
        """
        生成器函数：流式返回 Agent 的最终回复。
        """
        if not self.agent_executor:
            raise RuntimeError("Agent not initialized. Call update_phase() first.")

        print(f"\n[System] LLM Request Started for User {self.session_id[:8]}...") 

        try:
            # session_id 参数对于 RunnableWithMessageHistory 很重要，但在我们这个架构里
            # chat_history 已经绑定在 self 实例上了，这里传什么字符串其实不影响隔离，
            # 但为了规范，还是传入 session_id
            stream_iterable = self.agent_executor.stream(
                {"input": user_input},
                config={
                    "configurable": {"session_id": self.session_id},
                    "callbacks": callbacks or [] 
                }
            )

            for chunk in stream_iterable:
                # 只有当包含 "output" 时，才是最终给用户的回复
                if "output" in chunk:
                    # 打印一个小点，表示正在接收数据
                    sys.stdout.write(".") 
                    sys.stdout.flush()
                    yield chunk["output"]
            
            print(f"\n[System] Stream finished for User {self.session_id[:8]}.")

        except Exception as e:
            error_msg = f"System Error during execution: {str(e)}"
            print(f"\n[Error] {error_msg}")
            yield error_msg

    def chat(self, user_input: str):
        response = []
        for chunk in self.chat_stream(user_input):
            response.append(chunk)
        return "".join(response)

    def clear_short_term_memory(self):
        self.chat_history.clear()
        print(f"[System] Short-term conversation memory cleared for User {self.session_id[:8]}.")