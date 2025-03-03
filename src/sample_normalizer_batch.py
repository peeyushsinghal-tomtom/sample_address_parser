import pandas as pd
import yaml
import os
import json
import logging
from unidecode import unidecode
from openai import AzureOpenAI
from pathlib import Path
from tqdm import tqdm
from post_process_normalized_address import AddressPostProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SampleNormalizer:
    def __init__(self, config_path="src/config.yml"):
        logger.info("Initializing SampleNormalizer...")
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                logger.info("Successfully loaded config file")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise
        
        # Check for the Azure OpenAI endpoint
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            logger.error("AZURE_OPENAI_ENDPOINT environment variable is not set")
            raise ValueError("Missing Azure OpenAI endpoint. Please set the AZURE_OPENAI_ENDPOINT environment variable.")
        
        endpoint = endpoint.strip()
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        logger.info(f"Using Azure OpenAI endpoint: {endpoint}")

        # Check for the Azure OpenAI key
        api_key = os.getenv("AZURE_OPENAI_KEY")
        if not api_key:
            logger.error("AZURE_OPENAI_KEY environment variable is not set")
            raise ValueError("Missing Azure OpenAI API key. Please set the AZURE_OPENAI_KEY environment variable.")
        logger.info("Successfully loaded Azure OpenAI API key")

        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=self.config["azure"]["api_version"],
                azure_endpoint=endpoint,
            )
            logger.info("Successfully initialized Azure OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    def load_sample_data(self):
        """Load the sampling data using the configured path"""
        logger.info("Loading sample data...")
        try:
            file_path = self.config['sample_file']['sampling_data_path'].format(
                sampling_run_id=self.config['sampling_run_id']
            )
            logger.info(f"Attempting to load data from: {file_path}")
            df = pd.read_csv(file_path)
            df['country'] = df['country'].str.lower()
            logger.info(f"Successfully loaded {len(df)} rows of data")
            return df
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            raise
    
    def prepare_batch_input(self, rows):
        """Prepare a batch of inputs for LLM"""
        batch_data = []
        for _, row in rows.iterrows():
            input_data = {col: row[col] for col in self.config['columns_to_llm']}
            batch_data.append(input_data)
        return batch_data
    
    def get_llm_batch_response(self, batch_data):
        """Get normalized addresses for a batch of inputs from LLM"""
        prompt = f"""
        Please normalize the following batch of address information.
        For each address in the batch, return a JSON object with these keys: 
        {', '.join(self.config['output_json_keys'])}
        
        Input batch information:
        {json.dumps(batch_data, indent=2, ensure_ascii=False)}
        
        Return a JSON array containing one object per input address, maintaining the same order.
        Each object should have the specified keys.
        """
        max_tokens = min(self.config['azure']['max_tokens_batch'] * len(batch_data), self.config['azure']['token_limit'])
        response = self.client.chat.completions.create(
            model=self.config['azure']['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config['azure']['temperature'],
            max_tokens=max_tokens,  # Using larger token limit for batch
            response_format={"type": "json_object"}
        )
        
        # Extract the list of addresses from the response
        response_content = json.loads(response.choices[0].message.content)
        if "addresses" in response_content:
            return response_content["addresses"]
        else:
            logger.error("Unexpected response format: missing 'addresses' key")
            raise ValueError("LLM response is not in the expected format")
    
    def process_country_data(self, df, country_code, batch_size=100):
        """Process data for a specific country in batches"""
        # Filter data for the country
        country_df = df[df['country'] == country_code.lower()].copy()
        logger.info(f"Processing {len(country_df)} rows for country: {country_code}")
        
        # Process in batches
        normalized_addresses = []
        for start_idx in tqdm(range(0, len(country_df), batch_size), 
                            desc=f"Processing {country_code} in batches of {batch_size}"):
            end_idx = min(start_idx + batch_size, len(country_df))
            batch = country_df.iloc[start_idx:end_idx]
            
            try:
                batch_input = self.prepare_batch_input(batch)
                batch_response = self.get_llm_batch_response(batch_input)                
                # Ensure we got responses for all items in the batch
                if isinstance(batch_response, list):
                    normalized_addresses.extend(batch_response)
                else:
                    logger.error(f"Unexpected response format for batch {start_idx}-{end_idx}")
                    raise ValueError("LLM response is not a list")
            
            except Exception as e:
                logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
                logger.error(f"\nErrored Batch input: \n{batch_input}")
                # Continue to the next batch on error
                continue


                
        # Add normalized addresses to dataframe
        normalized_df = pd.DataFrame(normalized_addresses)
        combined_df = pd.merge(normalized_df, country_df, on=['country', 'sample_id', 'searched_query'], how='outer', indicator=True)

        return combined_df
    
    def save_country_data(self, df, country_code):
        """Save the processed data for a country"""
        output_dir = self.config['output_file']['output_dir'].format(
            sampling_run_id=self.config['sampling_run_id'],
            country_code=country_code
        )
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(
            output_dir,
            self.config['output_file']['output_file_name']
        )
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data for {country_code} to {output_path}")
    
    def process_all_countries(self):
        """Process and save data for all configured countries"""
        # Load the sample data
        df = self.load_sample_data()
        
        # Determine which countries to process
        if 'all' in [c.lower() for c in self.config['country_code']]:
            countries_to_process = df['country'].unique()
            logger.info("Processing all countries found in the dataset")
        else:
            countries_to_process = self.config['country_code']
            logger.info(f"Processing specified countries: {countries_to_process}")

        # Process each country
        for country_code in countries_to_process:
            try:
                logger.info(f"Starting processing for country: {country_code}")
                country_df = self.process_country_data(df, country_code.lower(), batch_size=self.config['batch_size'])
                self.save_country_data(country_df, country_code.lower())
                logger.info(f"Completed processing for country: {country_code}")
            except Exception as e:
                logger.error(f"Failed to process country {country_code}: {e}")
                continue

def main():
    logger.info("Starting batch sample normalization process...")
    
    # Test environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    
    logger.info("Checking environment variables...")
    logger.info(f"AZURE_OPENAI_ENDPOINT is {'set' if endpoint and endpoint.strip() else 'NOT SET'}")
    logger.info(f"AZURE_OPENAI_KEY is {'set' if api_key and api_key.strip() else 'NOT SET'}")

    try:
        normalizer = SampleNormalizer()
        logger.info("Successfully initialized SampleNormalizer")
        
        normalizer.process_all_countries()
        
    except Exception as e:
        logger.error(f"Failed to run normalization process: {e}")
        raise

        logger.info("Starting address post-processing...")
    
    try:
        processor = AddressPostProcessor()
        processor.process_all_countries()
        logger.info("Post-processing completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to complete post-processing: {e}")
        raise

if __name__ == "__main__":
    main()
