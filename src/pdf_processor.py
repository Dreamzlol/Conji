from pathlib import Path
from typing import List
import anthropic
from PyPDF2 import PdfReader
import json
import logging
import os
from .models import QAPair
from .config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and QA pair generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def read_pdf(self, file_path: Path) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                return ' '.join(page.extract_text() for page in reader.pages)
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {e}")
            raise

    def generate_qa_pairs(self, content: str) -> List[QAPair]:
        """Generate QA pairs from PDF content using Claude."""
        prompt = self._create_prompt(content)
        
        try:
            message = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system="You are a technical documentation expert. Create precise, practical Q&A pairs that accurately reflect the source material.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_data = json.loads(message.content[0].text.strip())
            return [QAPair(**pair) for pair in response_data]
            
        except Exception as e:
            logger.error(f"Failed to generate QA pairs: {e}")
            return []

    def _create_prompt(self, content: str) -> str:
        """Create the prompt for QA pair generation."""
        return f"""
        You are tasked with creating educational question-answer pairs from technical documentation. Focus exclusively on the content provided in the PDF.

        Instructions:
        1. Generate 20 question-answer pairs that cover key concepts, techniques, and code examples from the documentation
        2. For code-related content:
           - Use only code examples that appear in the documentation
           - Include the original code snippets in your answers
           - Explain the code's purpose and functionality
           - Format code using Markdown with the correct language identifier: ```language\\ncode\\n```
        3. For conceptual content:
           - Focus on technical definitions, processes, and important concepts
           - Reference specific sections from the documentation and always provide the example code from the PDF file
        4. Each answer should be detailed yet concise, focusing on practical understanding

        Format: Return a valid JSON array of objects with only the 'question' and 'answer' keys, without the JSON Markdown Code Block or any additional text.
        Content: {content}
        """ 