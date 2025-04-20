import sys
import os
import pandas as pd
from pandas._config import config
from sklearn.model_selection import train_test_split
from prepare_dataset import load_config
from src.loggers import get_logger

def data_split(config):
    logger = get_logger('DATA_SPLIT')

    # Загрузка данных из файла
    data_frame = pd.read_csv(config['featurize']['features_path'], sep=';')  # Убедитесь, что используется правильный разделитель

    # Разделение данных на обучающую и тестовую выборки
    train_dataset, test_dataset = train_test_split(
        data_frame,
        test_size=config['data_split']['test_size'],
        random_state=42
    )

    logger.info('Successfully split data into train and test sets')
    
    # Сохранение обучающего и тестового наборов
    train_csv_path = config['data_split']['trainset_path']
    test_csv_path = config['data_split']['testset_path']
    
    train_dataset.to_csv(train_csv_path, sep=';', index=False)  # Сохранение с правильным разделителем
    test_dataset.to_csv(test_csv_path, sep=';', index=False)

    logger.info(f'Train set saved to {train_csv_path}')
    logger.info(f'Test set saved to {test_csv_path}')

if __name__ == "__main__":
    # Загрузка конфигурации
    config = load_config("./src/config.yaml")
    
    # Выполнение разделения данных
    data_split(config)