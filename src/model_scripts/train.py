import pandas as pd
import numpy as np
import pickle
import mlflow
from mlflow.models import infer_signature

from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def eval_metrics(actual, pred, task="regression"):
    if task == "regression":
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return {"rmse": rmse, "mae": mae, "r2": r2}
    elif task == "classification":
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        auc = roc_auc_score(actual, pred)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

def train(config):
    # Load training and testing datasets
    df_train = pd.read_csv(config['data_split']['trainset_path'], sep=';')
    df_test = pd.read_csv(config['data_split']['testset_path'], sep=';')

    print(f"Training data shape: {df_train.shape}")
    print(f"Testing data shape: {df_test.shape}")

    # Select feature and target variables
    X_train = df_train.drop(columns=['Creditability'])
    y_train = df_train['Creditability']
    X_val = df_test.drop(columns=['Creditability'])
    y_val = df_test['Creditability']

    # Identify feature types
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Preprocessing steps
    transformers = []

    if config['featurize'].get('scale_features', True):
        transformers.append(('num', StandardScaler(), numeric_features))

    if config['featurize'].get('encode_categoricals', True) and categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))

    preprocessor = ColumnTransformer(transformers)

    # PowerTransformer (логгируем отдельно, как требует config)
    pt = PowerTransformer()
    X_train_scaled = pt.fit_transform(X_train[numeric_features])
    with open(config['train']['power_path'], "wb") as f:
        pickle.dump(pt, f)

    # Выбор модели
    model_type = config['train'].get('model_type', 'tree')
    param_grid = {}

    if model_type == "random_forest":
        param_grid = {
            "model__n_estimators": config["train"].get("n_estimators", [100]),
            "model__max_depth": config["train"].get("max_depth", [None]),
        }

    elif model_type == "logreg":
        param_grid = {
            "model__C": config["train"].get("C", [0.01, 0.1, 1.0]),
            "model__penalty": config["train"].get("penalty", ["l2"]),
        }

    elif model_type == "tree":
        param_grid = {
            "model__max_depth": config["train"].get("max_depth", [None]),
        }

    elif model_type == "sgd":
        param_grid = {
            "model__alpha": config["train"].get("alpha", [0.0001]),
            "model__fit_intercept": [True, False],
        }
    
    if model_type == "tree":
        model = DecisionTreeClassifier(random_state=config['train'].get('random_state', 42))
    elif model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=config['train'].get('random_state', 42))
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=config['train'].get('random_state', 42))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    mlflow.set_experiment("Creditability")
    with mlflow.start_run():
        print("MLflow run started!")
        # Обучение модели с GridSearchCV
        clf = GridSearchCV(pipeline, param_grid, cv=config['train']['cv'], n_jobs=4)
        clf.fit(X_train, y_train)

        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_val)

        # Метрики
        metrics = eval_metrics(y_val, y_pred, task="classification")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        print("Metrics:", metrics)
        