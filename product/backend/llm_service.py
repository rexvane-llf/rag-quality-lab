"""llm_service.py - LLM 服务集成（可选）"""
import os
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMService:
    """LLM 服务包装器"""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.client = None

        if self.provider and self.api_key:
            self._init_client()

    def _init_client(self):
        """初始化 LLM 客户端"""
        if self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider in ["openai", "deepseek"] and OPENAI_AVAILABLE:
            base_url = None
            if self.provider == "deepseek":
                base_url = "https://api.deepseek.com"
            self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return self.client is not None

    def generate_answer(self, question: str, context: str, model: Optional[str] = None) -> str:
        """生成答案"""
        if not self.is_available():
            return self._fallback_answer(context)

        prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请明确说明。

文档内容：
{context}

问题：{question}

请提供准确、简洁的回答："""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=model or "claude-3-sonnet-20240229",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            elif self.provider in ["openai", "deepseek"]:
                response = self.client.chat.completions.create(
                    model=model or "deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"LLM 生成失败: {e}")
            return self._fallback_answer(context)

    def _fallback_answer(self, context: str) -> str:
        """回退答案（无 LLM 时）"""
        return f"根据你的文档，以下是相关信息：\n\n{context[:500]}..."


# 全局 LLM 服务实例
llm_service = LLMService()
