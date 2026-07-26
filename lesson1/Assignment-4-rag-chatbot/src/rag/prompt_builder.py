"""
Prompt Builder for RAG Chatbot.
Constructs structured prompts with context injection, citation instructions, and tone guidelines.
"""

from typing import List, Optional
from src.core.types import SearchResult, ChatMessage


class RAGPromptBuilder:
    """Helper class to build RAG system and user prompts."""

    DEFAULT_SYSTEM_PROMPT = """You are an intelligent, helpful, and precise RAG (Retrieval-Augmented Generation) AI assistant.
Your goal is to answer user questions based on the provided Knowledge Base context.

Rules & Guidelines:
1. Grounding: Answer primarily using the provided context blocks. If the context contains the answer, explain it clearly and comprehensively.
2. Citations: Always cite the source titles or source numbers (e.g., [Source 1], [Source 2]) when using facts from the context.
3. Honesty: If the provided context does NOT contain enough information to answer the question, clearly state: "I couldn't find sufficient information in the provided knowledge base to answer this question," and offer general assistance if appropriate.
4. Tone: Maintain a professional, clear, and welcoming tone."""

    @classmethod
    def build_system_prompt(cls, custom_instructions: Optional[str] = None) -> str:
        """Return combined system prompt."""
        if custom_instructions:
            return f"{cls.DEFAULT_SYSTEM_PROMPT}\n\nAdditional Instructions:\n{custom_instructions}"
        return cls.DEFAULT_SYSTEM_PROMPT

    @classmethod
    def build_augmented_user_message(cls, query: str, formatted_context: str) -> str:
        """
        Inject retrieved knowledge base context into the user's query prompt.
        """
        augmented_prompt = (
            f"Please answer the question below using the following retrieved documents from our Knowledge Base:\n\n"
            f"<KNOWLEDGE_BASE_CONTEXT>\n{formatted_context}\n</KNOWLEDGE_BASE_CONTEXT>\n\n"
            f"User Question: {query}\n\n"
            f"Remember to cite relevant sources in your response."
        )
        return augmented_prompt

    @classmethod
    def prepare_chat_messages(
        cls,
        history: List[ChatMessage],
        current_query: str,
        formatted_context: str
    ) -> List[ChatMessage]:
        """
        Prepare full conversation message array with augmented context on the latest query.
        """
        messages = []
        # Include past turns
        for msg in history:
            messages.append(msg)
            
        # Append augmented current user query
        augmented_content = cls.build_augmented_user_message(current_query, formatted_context)
        messages.append(ChatMessage(role="user", content=augmented_content))
        return messages
