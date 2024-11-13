# Conji

# PDF to Synthetic Dataset

A Python tool that automatically generates high-quality question and answer pairs from PDF technical documentation using Claude AI.
Which can be used to create synthetic datasets for training LLMs.

## Features

- 📚 Batch processing of multiple PDF files
- 🤖 AI-powered Q&A pair generation using Claude 3.5
- 💾 JSONL output format for easy integration with ML pipelines
- 📊 Dataset statistics and progress tracking
- 🎨 Colorful console output with progress bars

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
- Generate Q&A pairs using Claude AI
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

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.