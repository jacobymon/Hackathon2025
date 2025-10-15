from collections import defaultdict, deque
from typing import Deque, Dict, List

# Simple in-memory conversation store (not for production).
# session_id -> deque of messages: {"role": "user"/"assistant"/"system", "content": "..."}
class ConversationManager:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.sessions: Dict[str, Deque[dict]] = defaultdict(lambda: deque([], maxlen=self.max_history))

    def append(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[dict]:
        return list(self.sessions[session_id])

    def reset(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

conv_manager = ConversationManager()
