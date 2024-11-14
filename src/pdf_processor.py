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
                system="""You are a specialized technical documentation analyzer. Your strict guidelines are:
1. NEVER create, modify, or combine code examples - use ONLY exact code snippets from the documentation
2. Extract and preserve code examples exactly as they appear in the source material, including comments and whitespace
3. Maintain precise technical terminology and definitions from the documentation
4. Generate questions that test understanding of the documented code examples and concepts
5. Every answer must be directly verifiable against the source material""",
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
        Analyze the provided technical documentation and generate question-answer pairs that precisely reflect its content.

        Critical Rules:
        - ONLY use code examples that appear verbatim in the documentation
        - DO NOT create new code examples or modify existing ones
        - DO NOT combine code snippets or create variations
        - If a concept doesn't have a code example in the documentation, do not provide one

        Core Requirements:
        1. Generate 20 question-answer pairs with the following distribution:
           - 60% questions about specific code examples from the documentation
           - 20% questions about documented concepts and their implementation
           - 20% questions about documented use cases and best practices

        2. Code Example Requirements:
           - Copy code snippets exactly as they appear in the documentation
           - Include the complete context as shown in the documentation
           - Preserve all original formatting, comments, and variable names
           - Format code using: ```language\\ncode\\n```
           - Reference the specific section/page where the code appears

        3. Technical Accuracy Requirements:
           - Use only terminology and definitions present in the documentation
           - Include exact version numbers and compatibility information
           - Reference specific documentation sections for all information
           - Never extrapolate or add information not in the source material

        4. Answer Structure:
           - Start with relevant quotes or references from the documentation
           - Include only code examples that appear in the source material
           - Explain concepts using the documentation's own terminology
           - List any limitations or requirements mentioned in the documentation

        Output Format: Return a JSON array containing objects with 'question' and 'answer' keys only. Do not include any additional formatting or explanation.

        Documentation Content: {content}
        """ 