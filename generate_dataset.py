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

def generate_qa_pairs(content):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("❌ ANTHROPIC_API_KEY not found in .env file")

    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""
    You are tasked with creating educational question-answer pairs from technical documentation. Focus exclusively on the content provided in the PDF.

    Instructions:
    1. Generate 20 question-answer pairs that cover key concepts, techniques, and code examples from the documentation
    2. For code-related content:
       - Use only code examples that appear in the documentation
       - Include the original code snippets in your answers
       - Explain the code's purpose and functionality
       - Format code using Markdown with the correct language identifier: ```language\ncode\n```
    3. For conceptual content:
       - Focus on technical definitions, processes, and important concepts
       - Reference specific sections from the documentation and always provide the example code from the PDF file
    4. Each answer should be detailed yet concise, focusing on practical understanding

    Format: Return a valid JSON array of objects with only the 'question' and 'answer' keys, without the JSON Markdown Code Block or any additional text.
    Content: {content}
    """
    
    logger.info("🤖 Generating Q&A pairs...")
    
    message = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=8192,
        temperature=0.3,
        system="You are a technical documentation expert. Create precise, practical Q&A pairs that accurately reflect the source material.",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    try:
        print(message.content[0].text)
        response_text = message.content[0].text.strip()
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"{Fore.RED}❌ Error processing response: {str(e)}{Fore.RESET}")
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