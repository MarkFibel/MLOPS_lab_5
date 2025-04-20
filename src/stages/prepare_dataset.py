import pandas as pd
import yaml
import sys
import os

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer

sys.path.append(os.getcwd())
from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clear_data(config):
    path2data = config['data_load']['dataset_csv']
    df = pd.read_csv(path2data, sep=';')

    # Определение колонок из конфига или используем дефолтные
    cat_columns = config['featurize'].get('categorical_columns', [
        'Payment_Status_of_Previous_Credit', 'Purpose', 'Sex_Marital_Status', 'Guarantors',
        'Type_of_apartment', 'Occupation', 'Telephone', 'Foreign_Worker'
    ])
    num_columns = config['featurize'].get('numeric_columns', [
        'Account_Balance', 'Duration_of_Credit_monthly', 'Credit_Amount',
        'Value_Savings_Stocks', 'Length_of_current_employment', 'Instalment_per_cent',
        'Age_years', 'Concurrent_Credits', 'No_of_Credits_at_this_Bank', 'No_of_dependents'
    ])

    # Очистка
    df = df.dropna()
    df = df[df['Credit_Amount'] >= 0]
    df = df[df['Duration_of_Credit_monthly'] > 0]
    df = df.reset_index(drop=True)

    # Кодирование категориальных признаков
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    return df

def scale_frame(frame, config):
    df = frame.copy()
    X = df.drop(columns=['Creditability'])
    y = df['Creditability']

    scaler = StandardScaler()
    power_trans = PowerTransformer()

    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))

    # Сохраняем трансформер
    with open(config['train']['power_path'], 'wb') as f:
        import pickle
        pickle.dump(power_trans, f)

    return X_scaled, y_scaled

def featurize(df, config) -> pd.DataFrame:
    logger = get_logger('FEATURIZE')
    logger.info('Create features')

    df['Relative_Credit_Amount'] = df['Credit_Amount'] / (df['Account_Balance'] + 1)
    df['Credit_per_month'] = df['Credit_Amount'] / (df['Duration_of_Credit_monthly'] + 1)
    df['Age_at_end_of_credit'] = df['Age_years'] + df['Duration_of_Credit_monthly'] // 12
    df['Credit_per_dependent'] = df['Credit_Amount'] / (df['No_of_dependents'] + 1)
    df['Instalment_to_income_ratio'] = df['Instalment_per_cent'] / (df['Account_Balance'] + 1)
    df['Stability_Index'] = df['Length_of_current_employment'] + df['Duration_in_Current_address']
    df['Has_Guarantor'] = (df['Guarantors'] != 1).astype(int)
    df['Has_Telephone'] = (df['Telephone'] == 1).astype(int)
    df['Is_Foreign_Worker'] = (df['Foreign_Worker'] == 2).astype(int)
    df['Credit_to_age_ratio'] = df['Credit_Amount'] / (df['Age_years'] + 1)

    output_path = config['featurize']['features_path']
    df.to_csv(output_path, sep=';', index=False)
    logger.info(f"Featurized data saved to {output_path}")
    return df

if __name__ == "__main__":
    config = load_config("./src/config.yaml")

    df_clean = clear_data(config)
    df_featurized = featurize(df_clean, config)

    if config.get('featurize', {}).get('scale_data', True):
        X_scaled, y_scaled = scale_frame(df_featurized, config)
        print("Data scaled successfully.")