# scheduler.py
import subprocess
import schedule
import time
from logger_setup import get_logger

logger = get_logger(__name__)

# Update these paths to match your environment
PYTHON_EXE = r"C:\Users\47936\OneDrive\Desktop\Portfolio Projects\.venv\Scripts\python.exe"
COLLECT_SCRIPT = r"C:\Users\47936\OneDrive\Desktop\Portfolio Projects\collect_data.py"
CLEAN_SCRIPT = r"C:\Users\47936\OneDrive\Desktop\Portfolio Projects\clean_data.py"
GOEMO_CLASSIFY_SCRIPT = r"C:\Users\47936\OneDrive\Desktop\Portfolio Projects\goemotions_classification.py"
MODEL_TRAINING_SCRIPT = r"C:\Users\47936\OneDrive\Desktop\Portfolio Projects\model_traning_goemotions.py"

def run_pipeline():
    logger.info("Running collect_data.py...")
    try:
        collect_proc = subprocess.run(
            [PYTHON_EXE, COLLECT_SCRIPT],
            capture_output=True, text=True
        )
        logger.info(collect_proc.stdout)
        if collect_proc.stderr:
            logger.error(f"Errors (collect_data): {collect_proc.stderr}")
    except Exception as e:
        logger.error(f"Failed to run {COLLECT_SCRIPT}: {e}", exc_info=True)

    logger.info("Running clean_data.py...")
    try:
        clean_proc = subprocess.run(
            [PYTHON_EXE, CLEAN_SCRIPT],
            capture_output=True, text=True
        )
        logger.info(clean_proc.stdout)
        if clean_proc.stderr:
            logger.error(f"Errors (clean_data): {clean_proc.stderr}")
    except Exception as e:
        logger.error(f"Failed to run {CLEAN_SCRIPT}: {e}", exc_info=True)

    logger.info("Running goemotions_classification.py...")
    try:
        class_proc = subprocess.run(
            [PYTHON_EXE, GOEMO_CLASSIFY_SCRIPT],
            capture_output=True, text=True
        )
        logger.info(class_proc.stdout)
        if class_proc.stderr:
            logger.error(f"Errors (goemotions_classification.py): {class_proc.stderr}")
    except Exception as e:
        logger.error(f"Failed to run {GOEMO_CLASSIFY_SCRIPT}: {e}", exc_info=True)

    logger.info("Pipeline run complete.\n")

def retrain_model():
    logger.info("Running model_training_goemotions.py for scheduled retraining...")
    try:
        train_proc = subprocess.run(
            [PYTHON_EXE, MODEL_TRAINING_SCRIPT],
            capture_output=True, text=True
        )
        logger.info(train_proc.stdout)
        if train_proc.stderr:
            logger.error(f"Errors (model_training_goemotions.py): {train_proc.stderr}")
    except Exception as e:
        logger.error(f"Failed to run {MODEL_TRAINING_SCRIPT}: {e}", exc_info=True)

# Run the pipeline every 2 minutes (just for testing)
schedule.every(2).minutes.do(run_pipeline)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)
