# train_model.py
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_evaluate(tune_hyperparams=False):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    preprocessed_path = os.path.join(project_root, "data/preprocessed_spotify_data.csv")  # read CSV directly
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(preprocessed_path)
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline models
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    results = []
    best_model = None
    best_score = -float("inf")
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append((name, r2, mae, rmse))
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    print(f"✅ Best baseline model: {best_model_name} (R²={best_score:.3f})")

    # Optional hyperparameter tuning
    if tune_hyperparams and best_model_name in ["RandomForest", "GradientBoosting"]:
        print("🔧 Running hyperparameter tuning...")
        if best_model_name == "RandomForest":
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
            grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
        else:  # GradientBoosting
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
            grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"✅ Hyperparameter-tuned best model: {best_model_name}")
        print(f"Best params: {grid.best_params_}")

    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved to: {model_path}")

    return sorted(results, key=lambda x: x[1], reverse=True)
