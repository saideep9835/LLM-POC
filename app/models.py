from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    provider: Optional[Literal["openai", "anthropic"]] = None  # Override default provider
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
                "provider": "openai"
            }
        }

class PerformanceMetrics(BaseModel):
    total_time_seconds: float
    first_token_time_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    response: str
    model: str
    provider: str
    usage: dict
    performance: PerformanceMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    version: str
    providers_available: List[str]
