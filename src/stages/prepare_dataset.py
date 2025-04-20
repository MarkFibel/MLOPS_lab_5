from pandas._config import config
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
import pandas as pd
import yaml
import sys
import os

sys.path.append(os.getcwd())
from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clear_data(path2data):
    df = pd.read_csv(path2data, sep=';')  # Изменение разделителя на ';'
    
    # Определение категориальных и числовых столбцов
    cat_columns = ['Payment_Status_of_Previous_Credit', 'Purpose', 'Sex_Marital_Status', 'Guarantors',
                   'Type_of_apartment', 'Occupation', 'Telephone', 'Foreign_Worker'] 
    num_columns = ['Account_Balance', 'Duration_of_Credit_monthly', 'Credit_Amount',
                   'Value_Savings_Stocks', 'Length_of_current_employment', 'Instalment_per_cent',
                   'Age_years', 'Concurrent_Credits', 'No_of_Credits_at_this_Bank', 'No_of_dependents']

    # Предварительная очистка данных
    # Например, удаление потенциально неверных значений (условия могут быть изменены)
    df = df.dropna()  # Удаление строк с пропущенными значениями
    
    # Здравый смысл и бизнес-правила, базовые проверки
    df = df[df['Credit_Amount'] >= 0]  # Удаляем кредиты с отрицательной суммой
    df = df[df['Duration_of_Credit_monthly'] > 0]  # Удаляем кредиты с невалидной продолжительностью
    
    df = df.reset_index(drop=True)
    
    # Кодирование категориальных признаков
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    return df

def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['Creditability']), df['Creditability']  # Признак качества кредита как целевая переменная
    scaler = StandardScaler()
    power_trans = PowerTransformer()

    # Масштабируем данные
    X_scale = scaler.fit_transform(X.values)  
    y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))

    return X_scale, y_scale, power_trans

def featurize(dframe, config) -> None:
    """
        Генерация новых признаков
    """
    logger = get_logger('FEATURIZE')
    logger.info('Create features')
    
    # Пример создания нового признака
    dframe['Relative_Credit_Amount'] = dframe['Credit_Amount'] / dframe['Account_Balance']
    
    # Создание пути для сохранения новых признаков
    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, sep=';', index=False)  # Сохраняем данные с разделителем ';'

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_prep = clear_data(config['data_load']['dataset_csv'])
    df_new_featur = featurize(df_prep, config)