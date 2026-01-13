from pydantic import BaseModel, Field
from typing import Optional

# ===============================
# 1. Chat Request Model
# ===============================
class ChatRequest(BaseModel):
  """
  Incoming user message.
  """

  user_query: str = Field(
    ...,
    min_length=3,
    max_length=1000,
    description="User question about loans, eligibility, or documents",
    example="What documents do I need to apply for a personal loan?"
  )
  request_id: Optional[str] = None

# ===============================
# 2. Chat Response Model
# ===============================
class ChatResponse(BaseModel):
  """
  AI-generated response.
  """

  answer: str = Field(
    ...,
    description="AI assistant response grounded in lending policy",
    example="To apply for a personal loan, you typically need proof of income..."
  )
  request_id: str