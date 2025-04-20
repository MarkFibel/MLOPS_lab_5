import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import pickle

def eval_metrics(actual, pred):
    """Calculate evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

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

    # Preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Create column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', SGDRegressor(random_state=42))
    ])

    # Set up MLflow experiment tracking
    mlflow.set_experiment("Creditability Prediction")
    with mlflow.start_run():
        params = {'model__alpha': config['train']['alpha'],
                  'model__fit_intercept': [False, True]}
        
        # Hyperparameter tuning with GridSearchCV
        clf = GridSearchCV(pipeline, params, cv=config['train']['cv'], n_jobs=4)
        clf.fit(X_train, y_train)

        # Best model prediction
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_val)

        # Evaluation metrics
        (rmse, mae, r2) = eval_metrics(y_val, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        print(f"R2: {r2}")

        # Log signature and model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Save the pipeline
        with open(config['train']['model_path'], "wb") as file:
            pickle.dump(best_model, file)