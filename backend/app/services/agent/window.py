from dataclasses import dataclass, field


@dataclass
class SlidingWindow:
    max_messages: int = 12
    messages: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            self.summary = f"此前对话摘要：{self.summary} {removed['content']}".strip()

    def context(self) -> list[dict[str, str]]:
        result = []
        if self.summary:
            result.append({"role": "system", "content": self.summary})
        return result + self.messages

