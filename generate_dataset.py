import os
import json
import anthropic
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from colorama import init, Fore
from alive_progress import alive_bar, config_handler
import logging
import re

# Load environment variables from .env file
load_dotenv()

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def clean_json_string(json_str):
    """Clean and sanitize JSON string to handle common issues."""
    import re
    
    # Remove any potential Unicode control characters
    json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
    
    # Fix invalid escape sequences
    def fix_escapes(match):
        s = match.group(0)
        if s in ('\\n', '\\r', '\\t', '\\b', '\\f', '\\"', '\\\\'):
            return s
        return '\\\\' + s[1]
    
    # Fix escape sequences
    json_str = re.sub(r'\\[^"\\\/bfnrt]', fix_escapes, json_str)
    
    # Fix delimiter issues
    json_str = re.sub(r'\}\s*\{', '},{', json_str)  # Fix missing commas between objects
    json_str = re.sub(r'\]\s*\[', '],[', json_str)  # Fix missing commas between arrays
    json_str = re.sub(r'(?<=[}\]])\s*(?=[{\[])', ',', json_str)  # Add missing commas
    
    # Remove trailing commas (invalid in JSON)
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Handle any remaining invalid escape sequences
    json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', json_str)
    
    try:
        # Verify the JSON is valid by parsing and re-stringifying it
        parsed = json.loads(json_str)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        # If parsing fails, try to fix common structural issues
        if not json_str.startswith('['):
            json_str = '[' + json_str
        if not json_str.endswith(']'):
            json_str = json_str + ']'
        
        # Add debug logging to see the problematic JSON
        logger.debug(f"Problematic JSON: {json_str[:1000]}...")
        return json_str

def generate_qa_pairs(content):
    # Get the API key from the environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("❌ ANTHROPIC_API_KEY not found in .env file")

    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""
    You are tasked with creating educational question-answer pairs from technical documentation. Focus exclusively on the content provided in the PDF.

    Instructions:
    1. Generate 25 question-answer pairs that cover key concepts, techniques, and code examples from the documentation
    2. For code-related content:
       - Use only code examples that appear in the documentation
       - Include the original code snippets in your answers
       - Explain the code's purpose and functionality
       - Format code using Markdown with the correct language identifier: ```language\ncode\n```
    3. For conceptual content:
       - Focus on technical definitions, processes, and important concepts
       - Reference specific sections from the documentation and always provide the example code from the PDF file
    4. Each answer should be detailed yet concise, focusing on practical understanding

    Format: ALWAYS return a valid JSON array of objects with 'question' and 'answer' keys, without the JSON Markdown Code Block. Make sure you return a valid JSON array.
    Content: {content}
    """
    
    logger.info("🤖 Generating Q&A pairs...")
    
    message = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=8192,
        temperature=0.3,
        system="You are a technical documentation expert. Create precise, practical Q&A pairs that accurately reflect the source material without adding external information. Focus on code examples, technical concepts, and implementation details exactly as presented in the documentation. Return ONLY a valid JSON array with no additional text or formatting.",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    try:
        # Get the response text and clean it
        response_text = message.content[0].text
        
        # Only remove JSON markdown blocks at the start/end of the response
        # This preserves code blocks within the answers
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json prefix
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ``` suffix
        response_text = response_text.strip()
        
        # Ensure the response starts with [ and ends with ]
        if not response_text.startswith('['):
            response_text = '[' + response_text
        if not response_text.endswith(']'):
            response_text = response_text + ']'
        
        cleaned_json = clean_json_string(response_text)
        
        # Try to parse the JSON with more detailed error handling
        try:
            qa_pairs = json.loads(cleaned_json)
        except json.JSONDecodeError as je:
            # Log the specific position where the error occurred
            logger.error(f"{Fore.RED}❌ JSON parsing error at position {je.pos}: {je.msg}{Fore.RESET}")
            logger.debug(f"Problem chunk: {cleaned_json[max(0, je.pos-50):min(len(cleaned_json), je.pos+50)]}")
            return []
            
        # Validate the structure
        if not isinstance(qa_pairs, list):
            logger.error(f"{Fore.RED}❌ Response is not a JSON array{Fore.RESET}")
            return []
            
        # Validate each QA pair
        valid_pairs = []
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            if 'question' not in pair or 'answer' not in pair:
                continue
            if not isinstance(pair['question'], str) or not isinstance(pair['answer'], str):
                continue
            valid_pairs.append(pair)
        
        if not valid_pairs:
            logger.error(f"{Fore.RED}❌ No valid QA pairs found in response{Fore.RESET}")
            return []
            
        return valid_pairs
        
    except Exception as e:
        logger.error(f"{Fore.RED}❌ Unexpected error processing response: {str(e)}{Fore.RESET}")
        logger.debug(f"Full response text: {response_text[:500]}...")  # Log first 500 chars
        return []

def save_to_jsonl(data, output_file):
    with open(output_file, 'a') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def main():
    data_folder = 'data'
    output_file = 'dataset.jsonl'
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    total_pdfs = len(pdf_files)
    
    print(f"\n{Fore.CYAN}📁 I found {total_pdfs} PDF files to process...{Fore.RESET}")
    
    # Keep only this progress bar
    with alive_bar(total_pdfs, bar='blocks', spinner='dots_waves', title='Processing PDFs') as bar:
        for filename in pdf_files:
            file_path = os.path.join(data_folder, filename)
            
            logger.info(f"🪄  Processing: {Fore.GREEN}{filename}{Fore.RESET}")
            
            try:
                content = read_pdf(file_path)
                logger.info(f"📖 Successfully read PDF: {Fore.GREEN}{filename}{Fore.RESET}")
                
                qa_pairs = generate_qa_pairs(content)
                logger.info(f"✨ Generated {len(qa_pairs)} Q&A pairs for {Fore.GREEN}{filename}{Fore.RESET}")
                
                # Create conversation pairs and save them
                for pair in qa_pairs:
                    conversation = {
                        "conversations": [
                            {"from": "human", "value": pair['question']},
                            {"from": "gpt", "value": pair['answer']}
                        ]
                    }
                    save_to_jsonl([conversation], output_file)
                
            except Exception as e:
                logger.error(f"{Fore.RED}❌ Error processing {filename}: {str(e)}{Fore.RESET}")
                continue
                
            bar()  # Update the progress bar
    
    print(f"\n{Fore.GREEN}✅ Dataset saved to {output_file}{Fore.RESET}")
    
    # Add dataset summary
    total_qa_pairs = 0
    with open(output_file, 'r') as f:
        for line in f:
            total_qa_pairs += 1
    
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total PDFs processed: {total_pdfs}")
    print(f"   • Total Q&A pairs generated: {total_qa_pairs}")
    print(f"   • Dataset file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main() 