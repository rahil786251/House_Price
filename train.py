import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "data/housing.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
TARGET_COLUMN = "Price"

def load_data(path):
    return pd.read_csv(path)

def get_feature_types(df, target_col):
    features = df.drop(columns=[target_col])
    numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

def make_model_pipelines(preprocessor):
    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1))
    ])
    return rf, xgb

def grid_search_model(pipeline, param_grid, X_train, y_train):
    gs = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    return gs

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    r2 = r2_score(y_test, preds)
    return {"rmse": rmse, "r2": r2}

def main():
    df = load_data(DATA_PATH)
    numeric_cols, categorical_cols = get_feature_types(df, TARGET_COLUMN)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    rf_pipe, xgb_pipe = make_model_pipelines(preprocessor)

    # Parameter grids
    rf_param_grid = {"model__n_estimators":[100,200], "model__max_depth":[None,10,20], "model__min_samples_split":[2,5]}
    xgb_param_grid = {"model__n_estimators":[100,200], "model__learning_rate":[0.05,0.1], "model__max_depth":[3,6]}

    print("Tuning RandomForest...")
    rf_gs = grid_search_model(rf_pipe, rf_param_grid, X_train, y_train)
    rf_eval = evaluate(rf_gs.best_estimator_, X_test, y_test)
    print("RF eval:", rf_eval)

    print("Tuning XGBoost...")
    xgb_gs = grid_search_model(xgb_pipe, xgb_param_grid, X_train, y_train)
    xgb_eval = evaluate(xgb_gs.best_estimator_, X_test, y_test)
    print("XGB eval:", xgb_eval)

    best_model = rf_gs.best_estimator_ if rf_eval["rmse"] <= xgb_eval["rmse"] else xgb_gs.best_estimator_
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

if __name__ == "__main__":
    main()
