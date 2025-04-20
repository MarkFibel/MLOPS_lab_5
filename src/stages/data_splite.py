import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from prepare_dataset import load_config
from src.loggers import get_logger

def data_split(config):
    logger = get_logger('DATA_SPLIT')

    # Путь к признаковому датасету
    features_path = config['featurize']['features_path']

    try:
        data_frame = pd.read_csv(features_path, sep=';')
        logger.info(f"Loaded data from {features_path}")
    except Exception as e:
        logger.error(f"Failed to read features CSV: {e}")
        raise

    # Разделение данных
    test_size = config['data_split'].get('test_size', 0.2)
    random_state = config['data_split'].get('random_state', 42)

    try:
        train_dataset, test_dataset = train_test_split(
            data_frame,
            test_size=test_size,
            random_state=random_state
        )
        logger.info(f"Data split completed: {len(train_dataset)} train / {len(test_dataset)} test")
    except Exception as e:
        logger.error(f"Error during data split: {e}")
        raise

    # Пути сохранения
    train_csv_path = config['data_split']['trainset_path']
    test_csv_path = config['data_split']['testset_path']

    try:
        train_dataset.to_csv(train_csv_path, sep=';', index=False)
        test_dataset.to_csv(test_csv_path, sep=';', index=False)
        logger.info(f"Train set saved to {train_csv_path}")
        logger.info(f"Test set saved to {test_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save datasets: {e}")
        raise

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    data_split(config)