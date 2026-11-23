from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class Question:
    question_id: str
    image_id: str
    template_id: str
    modality: str
    task_type: str
    prompt: str
    answer: float
    unit: str
    image_path: str
    metadata: dict[str, Any]

    def to_dict(self):
        return asdict(self)
