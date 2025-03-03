# Address Normalization Project

## Overview

This repo is designed to normalize address data using Azure OpenAI's language model. 
It processes input data, sends it to the model for normalization, and then post-processes the output to ensure data integrity and consistency.

## Features

- Load address data from CSV files.
- Normalize addresses using Azure OpenAI.
- Post-process the normalized data to merge duplicate entries.
- Save processed data to new CSV files.

## Requirements

- Python 3.7 or higher
- Required Python packages:
  - pandas
  - pyyaml
  - openai
  - unidecode
  - tqdm

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/streetnamematcher/address-normalization.git
   cd address-normalization
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a file named `set_env_variable.sh` and add your Azure OpenAI endpoint and key:
   ```bash
   #!/bin/bash
   export AZURE_OPENAI_ENDPOINT="your-endpoint"
   export AZURE_OPENAI_KEY="your-key"
   ```

4. **Configure the YAML file:**
   Edit `src/config.yml` to set the appropriate parameters for your data processing, including the sampling run ID, input file paths, and output configurations.

## Running the Scripts

1. **Run the normalizer:**
   To normalize the address data, execute the following command:
   ```bash
   ./run_normalizer.sh
   ```

2. **Post-process the normalized data:**
   After normalization, run the post-processing script to clean up the data:
   ```bash
   ./run_post_process.sh
   ```

## File Structure
```
├── src
│ ├── config.yml
│ ├── post_process_normalized_address.py
│ ├── sample_normalizer.py
│ └── sample_normalizer_batch.py
├── set_env_variable.sh
├── run_normalizer.sh
└── run_post_process.sh
```

## Notes

- Ensure that your Azure OpenAI credentials are correctly set in the `set_env_variable.sh` file.
- The output files will be saved in the specified directories as defined in the `config.yml` file.
- The post-processing script will merge duplicate entries based on the `sample_id` and update the `_merge` column accordingly.
