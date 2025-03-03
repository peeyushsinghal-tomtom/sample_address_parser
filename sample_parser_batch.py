import yaml
import pandas as pd
from typing import List, Dict
import os
import json
from openai import AzureOpenAI
import logging
from pathlib import Path
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv("/Users/peeyush.singhal/Library/CloudStorage/OneDrive-TomTom/projects/sample-address-parser/azure.env")

class AddressParserBatch:
    def __init__(self, config_path: str = "src/config_parser.yml"):
        """Initialize the address parser with configuration."""
        self.config = self._load_config(config_path)
        # self.client = AzureOpenAI(
        #     api_version=self.config['azure']['api_version'],
        #     azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        #     api_key=os.getenv('AZURE_OPENAI_KEY'),
        # )
        self.model = ChatCompletionsClient(
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            credential=AzureKeyCredential(os.getenv('AZURE_OPENAI_KEY')),
        )

        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                print(config) #TODO: Remove this
                return config
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def parse_addresses(self, 
                       input_file: str = None, 
                       column_name: str = None,
                       batch_size: int = None) -> pd.DataFrame:
        """
        Parse addresses in batches from the specified column.
        
        Args:
            input_file: Path to the input CSV file (optional, uses config if not provided)
            column_name: Name of the column containing addresses to parse (optional, uses config if not provided)
            batch_size: Number of addresses to process in each batch (optional, uses config if not provided)
        
        Returns:
            DataFrame with parsed address components
        """
        try:
            # Use provided parameters or fall back to config values
            input_file = input_file or self.config['file_name']
            column_name = column_name or self.config['input_column'][0]
            batch_size = batch_size or self.config['batch_size']

            # Read input file
            df = pd.read_csv(input_file)
            print(df.head()) #TODO: Remove this
            if column_name not in df.columns:
                raise ValueError(f"Column {column_name} not found in the input file")
            
            # Initialize empty lists to store results
            parsed_results = []
            
            # Process in batches
            for i in range(0, len(df), batch_size):
                batch = df[column_name].iloc[i:i+batch_size].tolist()
                parsed_batch = self._process_batch(batch)
                parsed_results.extend(parsed_batch)
                
                logger.info(f"Processed batch {i//batch_size + 1}")
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(parsed_results)
            
            # Combine with original DataFrame
            final_df = pd.concat([df, results_df], axis=1)
            
            # Create output directory if it doesn't exist
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            output_path = output_dir / self.config['output_file_name']
            final_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error processing addresses: {e}")
            raise

    def _process_batch(self, addresses: List[str]) -> List[Dict]:
        """Process a batch of addresses using the Azure OpenAI API."""
        try:
            # Prepare the prompt from config
            system_prompt = self.config['prompts']['system_prompt']
            user_prompt_template = self.config['prompts']['user_prompt']
            
            # Combine addresses into a single prompt
            addresses_text = "\n".join([f"Address {i+1}: {addr}" for i, addr in enumerate(addresses)])
            user_prompt = user_prompt_template.format(addresses=addresses_text)
            
            # Call Azure OpenAI
            response = self.model.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt)
                ],
                model=self.config['azure']['model'],
                model_extras={
                    "temperature": self.config['azure']['temperature'],
                    "max_tokens": self.config['azure']['max_tokens_batch'],
                }
            )
            
            # Parse the response
            parsed_results = self._parse_llm_response(response.choices[0].message.content, len(addresses))
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise

    def _parse_llm_response(self, response: str, expected_count: int) -> List[Dict]:
        """
        Parse the LLM response into structured data.
        
        Args:
            response: The response string from the LLM
            expected_count: Expected number of addresses in the response
            
        Returns:
            List of dictionaries containing parsed address components
        """
        try:
            # Initialize results list
            results = []
            
            # Expected components from config
            expected_components = self.config['components']
            
            # Split response into lines and process each address
            lines = response.strip().split('\n')
            current_address = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a new address
                if line.startswith('Address'):
                    if current_address:
                        results.append(current_address)
                        current_address = {}
                    continue
                
                # Parse component lines
                for component in expected_components:
                    if line.lower().startswith(f"{component.lower()}:"):
                        value = line.split(':', 1)[1].strip()
                        current_address[component] = value
                        break
            
            # Add the last address if exists
            if current_address:
                results.append(current_address)
            
            # Validate results
            if len(results) != expected_count:
                logger.warning(f"Expected {expected_count} addresses but got {len(results)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            raise

def main():
    parser = AddressParserBatch()
    
    result_df = parser.parse_addresses()

if __name__ == "__main__":
    main() 