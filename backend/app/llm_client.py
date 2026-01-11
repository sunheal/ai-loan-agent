import os
import logging
from typing import Optional, List, Dict

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LLMClient:
  """
  Wrapper around OpenAI's Chat Completions for consistent usage across:
  - LangGraph nodes
  - Retrieval validation
  - Query classification
  - Answer generation
  """

  def __init__(
    self,
    model: str = "gpt-4.1",
    temperature: float = 0.2,
    max_tokens: int = 500
  ):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
      raise ValueError("OPENAI_API_KEY not found in environment variables.")

    self.client = OpenAI(api_key=api_key)
    self.model = model
    self.temperature = temperature
    self.max_tokens = max_tokens

  def generate(
      self,
      messages: List[Dict[str, str]],
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
  ) -> str:
    """
    Sends messages to OpenAI Chat Completion.

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """

    try:
      response = self.client.chat.completions.create(
        model = self.model,
        messages = messages,
        temperature = temperature or self.temperature,
        max_tokens = max_tokens or self.max_tokens,
      )

      return response.choices[0].message.content

    except Exception as e:
      logger.error(f"[LLMClient] error during completion: {str(e)}")
      raise e

  def simple_query(
    self,
    prompt: str,
    system_prompt: Optional[str] = None
  ) -> str:
    """
    Helper for simple one-shot calls.
    """
    messages = []

    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    return self.generate(messages)