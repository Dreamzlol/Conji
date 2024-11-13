from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

@dataclass
class Config:
    """Application configuration settings."""
    data_folder: Path
    output_file: Path
    model_name: str
    max_tokens: int
    temperature: float

    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from environment and defaults."""
        load_dotenv()
        
        return cls(
            data_folder=Path('data'),
            output_file=Path('dataset.jsonl'),
            model_name="claude-3-5-haiku-latest",
            max_tokens=8192,
            temperature=0.3
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        if not os.getenv('ANTHROPIC_API_KEY'):
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
        if not self.data_folder.exists():
            raise ValueError(f"Data folder not found: {self.data_folder}") 