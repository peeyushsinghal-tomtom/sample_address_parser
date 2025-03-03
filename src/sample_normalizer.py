import pandas as pd
import yaml
import os
import json
import logging
from unidecode import unidecode
from openai import AzureOpenAI
from pathlib import Path
from tqdm import tqdm


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
            logger.info(f"Successfully loaded {len(df)} rows of data")
            return df
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            raise
    
    def prepare_llm_input(self, row):
        """Prepare the input for LLM from selected columns"""
        input_data = {col: row[col] for col in self.config['columns_to_llm']}
        return input_data
    
    def get_llm_response(self, input_data):
        """Get normalized address from LLM"""
        prompt = f"""
        Please normalize the following address information and return a JSON with these keys: 
        {', '.join(self.config['output_json_keys'])}
        
        Input information:
        {json.dumps(input_data, indent=2)}
        
        Return only the JSON response.
        """
        
        response = self.client.chat.completions.create(
            model=self.config['azure']['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config['azure']['temperature'],
            max_tokens=self.config['azure']['max_tokens'],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def process_country_data(self, df, country_code):
        """Process data for a specific country"""
        # Filter data for the country
        country_df = df[df['country'] == country_code.lower()].copy()
        print(f"Processing {len(country_df)} rows for country: {country_code}")
        # Process each row through LLM
        normalized_addresses = []
        for idx, row in tqdm(country_df.iterrows(), total=len(country_df), desc=f"Processing {country_code}"):
            input_data = self.prepare_llm_input(row)
            llm_response = self.get_llm_response(input_data)
            normalized_addresses.append(llm_response)

                
        # Add normalized addresses to dataframe
        normalized_df = pd.DataFrame(normalized_addresses)
        country_df['normalized_address'] = normalized_df['normalized_address']
        country_df['normalized_address_unidecode'] = normalized_df['normalized_address_unidecode']
        
        # return pd.DataFrame()
        return country_df
    
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
    
    def process_all_countries(self):
        """Process and save data for all configured countries"""
        # Load the sample data
        df = self.load_sample_data()

        # # Process each country except 'es', 'nl', 'us'
        # for country_code in df['country'].unique():  
        #     if country_code.lower() not in ['es', 'nl', 'us']:
        #         print(f"Processing country: {country_code}")
        #         country_df = self.process_country_data(df, country_code.lower())
        #         self.save_country_data(country_df, country_code.lower())
        
        # Process each country
        for country_code in self.config['country_code']: # ISO 3166-1 alpha-2
            print(f"Processing country: {country_code}")
            country_df = self.process_country_data(df, country_code.lower())
            self.save_country_data(country_df, country_code.lower())

def main():
    logger.info("Starting sample normalization process...")
    
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

if __name__ == "__main__":
    main()
