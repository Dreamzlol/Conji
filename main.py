from pathlib import Path
import logging
from typing import List
import json
from alive_progress import alive_bar
from colorama import init, Fore
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.models import DatasetSummary

# Initialize colorama
init()

def setup_logging() -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

def save_to_jsonl(conversations: List[dict], output_file: Path) -> None:
    """Save conversations to JSONL format."""
    with open(output_file, 'a') as f:
        for conv in conversations:
            json.dump(conv, f)
            f.write('\n')

def get_dataset_summary(config: Config) -> DatasetSummary:
    """Generate summary statistics for the dataset."""
    total_qa_pairs = sum(1 for _ in open(config.output_file))
    return DatasetSummary(
        total_pdfs=len(list(config.data_folder.glob('*.pdf'))),
        total_qa_pairs=total_qa_pairs,
        file_size_mb=config.output_file.stat().st_size / (1024 * 1024)
    )

def print_summary(summary: DatasetSummary) -> None:
    """Print dataset summary statistics."""
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total PDFs processed: {summary.total_pdfs}")
    print(f"   • Total Q&A pairs generated: {summary.total_qa_pairs}")
    print(f"   • Dataset file size: {summary.file_size_mb:.2f} MB")

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate configuration
        config = Config.load()
        config.validate()
        
        # Initialize processor
        processor = PDFProcessor(config)
        
        # Get PDF files
        pdf_files = list(config.data_folder.glob('*.pdf'))
        print(f"\n{Fore.CYAN}📁 Found {len(pdf_files)} PDF files to process...{Fore.RESET}")
        
        # Process PDFs
        with alive_bar(len(pdf_files), bar='blocks', spinner='dots_waves', title='Processing PDFs') as bar:
            for pdf_path in pdf_files:
                logger.info(f"🪄  Processing: {Fore.GREEN}{pdf_path.name}{Fore.RESET}")
                
                try:
                    content = processor.read_pdf(pdf_path)
                    qa_pairs = processor.generate_qa_pairs(content)
                    
                    conversations = [pair.to_conversation() for pair in qa_pairs]
                    save_to_jsonl(conversations, config.output_file)
                    
                    logger.info(f"✨ Generated {len(qa_pairs)} Q&A pairs for {Fore.GREEN}{pdf_path.name}{Fore.RESET}")
                    
                except Exception as e:
                    logger.error(f"{Fore.RED}❌ Error processing {pdf_path.name}: {str(e)}{Fore.RESET}")
                    continue
                
                bar()
        
        print(f"\n{Fore.GREEN}✅ Dataset saved to {config.output_file}{Fore.RESET}")
        print_summary(get_dataset_summary(config))
        
    except Exception as e:
        logger.error(f"{Fore.RED}❌ Application error: {str(e)}{Fore.RESET}")
        raise

if __name__ == "__main__":
    main() 