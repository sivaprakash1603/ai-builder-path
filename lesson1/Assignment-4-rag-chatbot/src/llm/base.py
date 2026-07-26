"""
Abstract base interface for Large Language Model (LLM) generation providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from src.core.types import ChatMessage


class BaseLLMProvider(ABC):
    """Abstract interface for LLM text generation and chat completion."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate response from conversation messages.
        Returns dict: {"text": str, "model": str, "provider": str, "usage": Dict[str, int]}
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier string."""
        pass

    @property
    @abstractmethod
    def current_model(self) -> str:
        """Return the currently active model deployment name."""
        pass
