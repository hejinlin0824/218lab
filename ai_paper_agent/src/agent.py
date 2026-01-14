import sys
import os
from typing import Literal, List
from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

# 引入历史记录管理
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 引入向量检索相关库 (用于科研笔记持久化)
from langchain_community.vectorstores import FAISS
# === 修改点：新增 ChatOpenAI 导入，用于动态初始化 ===
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# === 修改点：不再从 config 导入 llm 和 API KEY，只导入路径 ===
from src.config import RES_DIR
from src.tools import ToolFactory
from src.prompts import PromptManager

class ResearchAgent:
    """
    科研Agent核心类。
    支持多研究员并发，每个实例绑定一个用户名作为 session_id。
    集成 FAISS 硬盘持久化记忆，支持基于方案A的笔记同步。
    """
    
    # === 修改点：初始化接收用户动态配置 ===
    def __init__(self, session_id: str, api_key: str, base_url: str, model: str):
        """
        初始化 Agent 实例。
        :param session_id: 用户的用户名
        :param api_key: 用户提供的 API Key
        :param base_url: 用户提供的 Base URL
        :param model: 用户选择的模型
        """
        self.session_id = session_id
        
        # 1. 动态构建研究员专属目录 (res/{username})
        self.session_dir = RES_DIR / self.session_id
        if not self.session_dir.exists():
            self.session_dir.mkdir(parents=True, exist_ok=True)
            
        # === 修改点：动态构建 LLM ===
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0, # 科研任务保持严谨
            api_key=api_key,
            base_url=base_url,
            streaming=True
        )
        
        # 2. 初始化嵌入模型 (用于 FAISS)
        # 注意：这里假设用户提供的 API Key 也支持 Embedding (通常 OpenAI/DeepSeek 格式兼容)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=base_url
        )
        
        # 3. 尝试从硬盘加载该用户的 FAISS 索引
        self.vector_store_path = self.session_dir / "faiss_index"
        self.vector_store = self._load_vector_store()
        
        # 4. 使用工厂生成绑定了特定路径的工具集
        self.tools = ToolFactory(self.session_dir).get_tools()
        
        # 5. 初始对话历史
        self.chat_history = ChatMessageHistory()
        self.agent_executor = None

    def _load_vector_store(self):
        """从硬盘加载持久化向量库"""
        if self.vector_store_path.exists():
            try:
                return FAISS.load_local(
                    str(self.vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True # 内部使用，确保安全
                )
            except Exception as e:
                print(f"[System] Warning: Failed to load vector store for {self.session_id}: {e}")
        return None

    def sync_knowledge_base(self) -> str:
        """
        方案A：手动触发同步。扫描用户目录下所有 .md 文件并持久化到 FAISS 硬盘索引。
        """
        all_docs = []
        # 扫描用户根目录下所有 Markdown 文件
        md_files = list(self.session_dir.glob("**/*.md"))
        
        if not md_files:
            return "没有找到任何可同步的 Markdown 笔记。"

        try:
            from langchain_community.document_loaders import TextLoader
            for md_path in md_files:
                # === 过滤逻辑：排除系统文件 ===
                if md_path.name == "memory.md":
                    continue
                # 排除以 . 开头的隐藏文件夹 (如 .silverbullet)
                if any(part.startswith('.') for part in md_path.parts):
                    continue
                # 排除 SilverBullet 插件库
                if "_plug" in md_path.parts or "Library" in md_path.parts:
                    continue
                # === 过滤结束 ===

                try:
                    loader = TextLoader(str(md_path), encoding='utf-8')
                    all_docs.extend(loader.load())
                except Exception as load_err:
                    print(f"[Warning] Failed to load {md_path}: {load_err}")
                    continue

            if not all_docs:
                return "未找到有效的笔记文件 (已忽略系统文件)。"

            # 文本切片
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(all_docs)

            # 构建并保存索引
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            self.vector_store.save_local(str(self.vector_store_path))
            
            return f"同步成功！已索引 {len(all_docs)} 个有效笔记文件，知识库已更新。"
        except Exception as e:
            return f"知识库同步失败: {str(e)}"

    def _build_agent(self, system_prompt_content: str):
        """构建底层 Agent 执行链"""
        # 如果存在向量库，则在 System Prompt 中注入检索提示
        if self.vector_store:
            system_prompt_content += "\n\n[Context: 你已连接到研究员的个人知识库，可以参考其过往笔记进行推导。]"

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
        """切换 Agent 的思考阶段，并递归加载前序文件作为上下文"""
        user_prefix = f"[User {self.session_id}]"
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
                # 从用户的 session_dir 读取前序文件
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

        print(f"\n[System] LLM Request Started for User {self.session_id}...") 

        try:
            # session_id 参数确保对话历史的隔离
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
            
            print(f"\n[System] Stream finished for User {self.session_id}.")

        except Exception as e:
            error_msg = f"System Error during execution: {str(e)}"
            print(f"\n[Error] {error_msg}")
            yield error_msg

    def chat(self, user_input: str):
        """同步聊天接口"""
        response = []
        for chunk in self.chat_stream(user_input):
            response.append(chunk)
        return "".join(response)

    def clear_short_term_memory(self):
        """清空短期对话缓存"""
        self.chat_history.clear()
        print(f"[System] Short-term conversation memory cleared for User {self.session_id}.")