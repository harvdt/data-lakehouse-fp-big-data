import os
import time
import logging
import hashlib
from datetime import datetime
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainingCoordinator:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.gold_path = os.path.join(self.base_path, "src/data/gold")
        self.model_path = os.path.join(self.base_path, "src/models/training")
        self.state_file = os.path.join(self.model_path, "training_state.json")
        self.min_interval = 90  # default crontab 5 menit sekali

        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Gold path: {self.gold_path}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"State file: {self.state_file}")
        
    def get_data_hash(self):
        """Calculate hash of gold data state"""
        hash_md5 = hashlib.md5()
        
        try:
            for parquet_file in Path(self.gold_path).glob("*_parquet"):
                if parquet_file.is_dir():
                    for file in parquet_file.glob("**/*"):
                        if file.is_file():
                            hash_md5.update(str(file.stat().st_mtime).encode())
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating data hash: {e}")
            return None
        
    def load_state(self):
        """Load the previous training state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        return {
            'last_training_time': 0,
            'last_data_hash': None,
            'training_in_progress': False
        }
        
    def save_state(self, state):
        """Save the current training state"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    def run_training(self):
        """Run the model training if needed"""
        try:
            state = self.load_state()
            
            if state['training_in_progress']:
                logger.info("Training already in progress, skipping...")
                return
            
            current_time = time.time()
            time_since_last = current_time - state['last_training_time']
            
            current_hash = self.get_data_hash()
            if current_hash is None:
                logger.warning("Could not calculate data hash, skipping training")
                return
            
            if (time_since_last < self.min_interval and 
                current_hash == state['last_data_hash']):
                logger.info("No new data and minimum interval not met, skipping training")
                return
            
            state['training_in_progress'] = True
            self.save_state(state)
            
            try:
                logger.info("Starting model training...")
                training_script = os.path.join(self.model_path, "model_training.py")
                if not os.path.exists(training_script):
                    logger.error(f"Training script not found at {training_script}")
                    return
                
                result = os.system(f"python3 {training_script}")
                if result == 0:
                    logger.info("Training completed successfully")
                    state.update({
                        'last_training_time': current_time,
                        'last_data_hash': current_hash,
                        'training_in_progress': False
                    })
                else:
                    logger.error("Training failed with exit code: %d", result)
                    state['training_in_progress'] = False
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                state['training_in_progress'] = False
                
            finally:
                self.save_state(state)
                
        except Exception as e:
            logger.error(f"Error in run_training: {e}")

def main():
    """Main function for running the coordinator"""
    coordinator = ModelTrainingCoordinator()
    
    while True:
        try:
            coordinator.run_training()
            # Sleep for 1 minute before next check
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Training coordinator stopped")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Sleep on error to prevent rapid retries

if __name__ == "__main__":
    main()