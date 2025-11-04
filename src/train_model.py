from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor()
    }

    results = []
    best_model = None
    best_score = -float("inf")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        results.append((name, r2, mae, rmse))

        if r2 > best_score:
            best_score = r2
            best_model = model

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)

    print(f"✅ Best model saved: {model_path}")
    return sorted(results, key=lambda x: x[1], reverse=True)  # Sort by best R²
