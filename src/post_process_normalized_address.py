import pandas as pd
import yaml
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AddressPostProcessor:
    def __init__(self, config_path="src/config.yml"):
        logger.info("Initializing AddressPostProcessor...")
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                logger.info("Successfully loaded config file")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise

    def load_country_data(self, country_code):
        """Load the normalized data file for a specific country"""
        try:
            file_path = os.path.join(
                self.config['output_file']['output_dir'].format(
                    sampling_run_id=self.config['sampling_run_id'],
                    country_code=country_code
                ),
                self.config['output_file']['output_file_name']
            )
            
            logger.info(f"Loading data for country {country_code} from {file_path}")
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.error(f"Failed to load data for country {country_code}: {e}")
            raise

    def process_country_data(self, df):
        """Process and clean the country data"""
        logger.info("Processing country data...")
        
        # Ensure country is lowercase
        df['country'] = df['country'].str.lower()
        
        # Handle rows where _merge is not 'both'
        if '_merge' in df.columns:
            logger.info(f"Found {len(df[df['_merge'] != 'both'])} rows with _merge != 'both'")
            
            # Create a new dataframe to store processed rows
            processed_df = pd.DataFrame()
            
            # Group by sample_id
            for sample_id, group in df.groupby('sample_id'):
                if len(group) > 1:
                    # For each column, take the first non-null value
                    merged_series = group.bfill().iloc[0]
                    merged_series['_merge'] = 'post_processed'
                    processed_df = pd.concat([processed_df, pd.DataFrame([merged_series])], ignore_index=True)
                else:
                    # Single row, just update _merge value
                    group['_merge'] = 'post_processed'
                    processed_df = pd.concat([processed_df, group], ignore_index=True)
            
            logger.info(f"Processed data now has {len(processed_df)} rows")
            return processed_df
        else:
            logger.warning("No '_merge' column found in the data")
            return df

    def save_country_data(self, df, country_code):
        """Save the processed data back to file"""
        try:
            output_dir = self.config['output_file']['output_dir'].format(
                sampling_run_id=self.config['sampling_run_id'],
                country_code=country_code
            )
            
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Add '_processed' suffix to the filename
            filename_parts = os.path.splitext(self.config['output_file']['output_file_name'])
            processed_filename = f"{filename_parts[0]}_processed{filename_parts[1]}"
            
            output_path = os.path.join(output_dir, processed_filename)
            
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data for country {country_code}: {e}")
            raise

    def process_all_countries(self):
        """Process all countries specified in the config"""
        logger.info("Starting post-processing for all countries...")
        
        for country_code in self.config['country_code']:
            try:
                country_code = country_code.lower()
                logger.info(f"Processing country: {country_code}")
                
                # Load data
                df = self.load_country_data(country_code)
                
                # Process data
                processed_df = self.process_country_data(df)
                
                # Save processed data
                self.save_country_data(processed_df, country_code)
                
                logger.info(f"Completed processing for country: {country_code}")
                
            except Exception as e:
                logger.error(f"Failed to process country {country_code}: {e}")
                continue

def main():
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