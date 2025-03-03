# Address Parser

A batch processing tool that uses Azure OpenAI to parse addresses into structured components.

## Overview

This tool takes addresses from a CSV file and uses Azure OpenAI's language models to parse them into components like house number, road, city, etc. It processes addresses in batches for efficiency and saves the results to a new CSV file.

## Prerequisites

- Python 3.8+
- Azure OpenAI API access
- Required Python packages (install via `pip install -r src/requirements.txt`):
  - pandas
  - pyyaml
  - python-dotenv
  - azure-ai-inference
  - openai

## Configuration

### Environment Variables

Create an `azure.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_KEY=your_api_key
```

### Configuration File

Create a `config_parser.yml` file with your configuration:

yaml
input_column:
normalized_address # Column name containing addresses to parse
components:
house_number
road
city
postcode
state
country
File paths
file_name: "path/to/input/file.csv"
output_dir: "path/to/output/directory/"
output_file_name: "parsed_address.csv"
Azure OpenAI settings
azure:
api_version: "2024-02-15-preview"
max_tokens: 300
max_tokens_few_shot: 1000
max_tokens_batch: 50000
model: "gpt-4o-mini"
temperature: 0.1
token_limit: 15000
batch_size: 50 # Number of addresses to process in each batch
Prompts for the model
prompts:
system_prompt: "You are an expert address parser. Parse the given addresses into their components."
user_prompt: "Please parse the following addresses into their components:\n\n{addresses}\n\nProvide the results in a structured format."
```

## Usage

1. Install dependencies:

```bash
pip install -r src/requirements.txt
```

2. Set environment variables:

```bash
source azure.env
```

3. Run the script:

```bash
python src/sample_parser_batch.py
```

## Output


The script will:
- Read addresses from the configured input CSV file
- Process them in batches through Azure OpenAI
- Parse the responses into structured components
- Save the results to the configured output location

## Output

The parser creates a new CSV file containing:
- All original columns from the input file
- Additional columns for each parsed address component (house_number, road, city, etc.)

## Error Handling

The script includes comprehensive error handling and logging:
- Configuration loading errors
- File access issues
- API communication errors
- Response parsing problems

Logs are written to stdout with appropriate error levels (INFO, WARNING, ERROR).
