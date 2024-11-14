![Group 1](https://github.com/user-attachments/assets/9fc91bcd-bfd2-4933-b966-e70335c28467)

## Features

- 📚 Batch processing of multiple PDF files
- 🤖 Generate Q&A pairs using Anthropic
- 💾 JSONL output format for easy integration with ML pipelines
- 📊 Dataset statistics and progress tracking

## Prerequisites

- Python 3.8+
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dreamzlol/Conji.git
cd Conji
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

1. Place your PDF files in the `data` folder
2. Run the script:
```bash
python main.py
```

The script will:
- Process all PDFs in the data folder
- Generate Q&A pairs using Anthropic
- Save the results in JSONL format
- Display progress and summary statistics

## Output Format

The generated dataset is saved in JSONL format with the following structure (Qwen chat template):
```json
{
    "conversations": [
        {"from": "human", "value": "question"},
        {"from": "gpt", "value": "answer"}
    ]
}
```

## Configuration

Key settings can be modified in `src/config.py`:
- Model name (default: "claude-3-5-haiku-latest")
- Maximum tokens (default: 8192)
- Temperature (default: 0.3)
- Input/output paths

## Project Structure

```
├── data/                 # PDF files directory
├── src/
│   ├── config.py         # Configuration settings
│   ├── models.py         # Data models
│   └── pdf_processor.py  # PDF processing and QA generation
├── main.py               # Main script
└── requirements.txt      # Dependencies
```

## Error Handling

- The script includes comprehensive error handling for PDF processing and API calls
- Failed PDF processing won't stop the entire batch
- Detailed error logging for debugging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
