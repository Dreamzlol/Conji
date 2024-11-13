from dataclasses import dataclass
from typing import List, Dict

@dataclass
class QAPair:
    """Represents a question-answer pair."""
    question: str
    answer: str

    def to_conversation(self) -> Dict:
        """Convert QA pair to conversation format."""
        return {
            "conversations": [
                {"from": "human", "value": self.question},
                {"from": "gpt", "value": self.answer}
            ]
        }

@dataclass
class DatasetSummary:
    """Summary statistics for the generated dataset."""
    total_pdfs: int
    total_qa_pairs: int
    file_size_mb: float 