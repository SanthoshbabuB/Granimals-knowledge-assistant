from langchain_core.chat_history import InMemoryChatMessageHistory


class ConversationMemory:
    """Store conversation history by conversation ID."""

    def __init__(self) -> None:
        self._histories: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(
        self,
        conversation_id: str,
    ) -> InMemoryChatMessageHistory:
        """Get existing history or create a new conversation."""
        if conversation_id not in self._histories:
            self._histories[conversation_id] = InMemoryChatMessageHistory()

        return self._histories[conversation_id]

    def add_user_message(
        self,
        conversation_id: str,
        message: str,
    ) -> None:
        """Store a user message."""
        history = self.get_history(conversation_id)
        history.add_user_message(message)

    def add_ai_message(
        self,
        conversation_id: str,
        message: str,
    ) -> None:
        """Store an assistant message."""
        history = self.get_history(conversation_id)
        history.add_ai_message(message)

    def get_messages(
        self,
        conversation_id: str,
    ) -> list:
        """Return conversation messages."""
        history = self.get_history(conversation_id)
        return history.messages