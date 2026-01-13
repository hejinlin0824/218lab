import os
import logging
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

# 加载环境变量 (作为默认配置的回退选项)
load_dotenv()

# 配置模块级日志
logger = logging.getLogger(__name__)

class LLMClient:
    """
    LLM API 客户端封装类。
    
    并发支持说明：
    为了支持多用户并发且互不干扰，__init__ 方法现在接收显式的 api_key 和 base_url。
    优先级逻辑：传入参数 > 环境变量 > 报错。
    """

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化 LLM 客户端。

        Args:
            api_key (str, optional): 用户的 API Key。如果未提供，尝试从 env 读取。
            base_url (str, optional): 自定义 API 地址。如果未提供，尝试从 env 读取。
            model (str, optional): 默认模型名称。
        """
        # 1. 确定配置优先级：参数 > 环境变量
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")
        self.default_model = model if model else os.getenv("LLM_MODEL", "gpt-4o")

        # 2. 安全校验
        if not self.api_key:
            logger.critical("API Key is missing. Please provide it via argument or .env file.")
            raise ValueError("API Key is required to initialize LLMClient.")

        # 3. 初始化 OpenAI 客户端实例
        # 注意：每个 LLMClient 实例现在持有自己独立的 connection
        if self.base_url:
            logger.info(f"Initializing Client with Custom Base URL: {self.base_url}")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            logger.info("Initializing Client with Default OpenAI Base URL")
            self.client = OpenAI(api_key=self.api_key)

    def get_completion(self, system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.3) -> str:
        """
        发送请求给 LLM 并获取响应。

        Args:
            system_prompt (str): 系统指令。
            user_prompt (str): 用户输入。
            model (str, optional): 本次请求特定的模型 (覆盖默认值)。
            temperature (float, optional): 采样温度。

        Returns:
            str: LLM 返回的文本内容。
        """
        # 确定本次请求使用的模型
        target_model = model if model else self.default_model
        
        logger.info(f"Sending request to LLM (Model: {target_model})...")

        try:
            # 4. 发起 API 调用
            response = self.client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )

            # 5. 提取结果
            content = response.choices[0].message.content
            
            # 记录 Token 消耗日志
            if response.usage:
                total = response.usage.total_tokens
                logger.info(f"Request successful. Total tokens used: {total}")
            
            return content

        except RateLimitError:
            logger.error("LLM Rate Limit Exceeded.")
            raise
        except APIConnectionError:
            logger.error("Failed to connect to LLM API.")
            raise
        except APIError as e:
            logger.error(f"LLM API returned an error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM call: {str(e)}")
            raise