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
        """Create an enhanced prompt for documentation QA pair generation."""
        return f'''
        You are an expert in creating educational content for programming languages. Analyze the provided technical documentation and generate diverse, high-quality question-answer pairs that will help train an LLM to understand and explain this technology.

        DOCUMENTATION ANALYSIS REQUIREMENTS:
        1. First analyze the documentation for:
        - Core concepts and fundamental principles
        - Specific syntax and usage patterns
        - Code examples and their context
        - Best practices and recommendations
        - Common use cases and scenarios
        - Technical limitations and edge cases

        QUESTION GENERATION GUIDELINES:
        Generate 10 questions across these categories:
        1. Conceptual Understanding (25%)
        - Definition and purpose questions
        - How concepts relate to each other
        - Technical terminology explanation
        Example: "What is the purpose of the $bindable rune in Svelte?"

        2. Practical Implementation (35%)
        - Syntax usage questions
        - Code pattern questions
        - Implementation steps
        Example: "How do you declare a bindable prop in a Svelte component?"

        3. Code Analysis (25%)
        - Code example interpretation
        - Syntax explanation
        - Output prediction
        Example: "What will happen in this code when the input value changes?"

        4. Best Practices & Edge Cases (15%)
        - Usage recommendations
        - Common pitfalls
        - Limitations and constraints
        Example: "When should you avoid using bindable props in Svelte?"

        ANSWER REQUIREMENTS:
        1. Include code examples ONLY if they appear verbatim in the documentation:
        ```language
        [exact code from documentation]
        ```

        3. Provide context about:
        - Where this information appears in the docs
        - How it relates to other concepts
        - Any important prerequisites

        4. Reference specific limitations, warnings, or notes from the docs

        QUALITY CONTROLS:
        - Never invent or modify code examples
        - Use only terminology present in the documentation
        - Maintain technical accuracy with source material
        - Include version-specific information when present
        - Preserve all original formatting and syntax

        OUTPUT FORMAT:
        Return a JSON array containing objects with 'question' and 'answer' keys only. Do not include any additional formatting or explanation.

        Documentation Content to Analyze: 
        {content}
        '''