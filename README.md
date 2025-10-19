# House Price Predictor

Simple project that trains a regression model to predict house prices and serves predictions through a Flask web app.

## Structure
- `data/housing.csv` - dataset (place your CSV here)
- `train.py` - training pipeline, model selection, exports `models/best_model.joblib`
- `app.py` - Flask web app for predictions
- `templates/index.html` - web UI
- `models/` - saved model and diagnostics
- `sample_generate_data.py` - optional script to generate synthetic dataset for testing

## Quick start
1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare data:
   - Put your CSV at `data/housing.csv`.
   - Ensure the target column is named `Price` (or edit `TARGET_COLUMN` in `train.py`).

   Or generate synthetic data to test:
   ```
   python sample_generate_data.py
   ```

4. Train:
   ```
   python train.py
   ```
   This will tune models, pick the best, and save `models/best_model.joblib`.

5. Run the app:
   ```
   python app.py
   ```
   Open `http://localhost:5000/` and try predictions.

## Notes & next steps
- Expand hyperparameter grids in `train.py` for better performance.
- Add cross-validation folds or use `RandomizedSearchCV` for faster hyperparameter search.
- Add feature importance reporting and SHAP explanations.
- Containerize with Docker and deploy to a cloud provider.
- Add unit tests for preprocessing and endpoints.
