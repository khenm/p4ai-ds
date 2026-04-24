import sys
import os

# Ensure src module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess.tabular_preprocess import load_and_preprocess
from src.models.salary.train import initialize_models, train_models
from src.models.salary.evaluate import evaluate_models

def main():
    print("="*50)
    print("Starting Job Salary Prediction ML Pipeline")
    print("="*50)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'jobsalary', 'job_salary_prediction_dataset.csv')
    results_dir = os.path.join(base_dir, 'ui', 'assets', 'data', 'jobsalary_ml')
    mapping_path = os.path.join(results_dir, 'preprocessing_mappings.json')

    print("\n[1] Applying Preprocessing (Ordinal & Target Encoding)...")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path, export_mapping_path=mapping_path)
    print(f"    Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"    Number of features: {X_train.shape[1]}")

    print("\n[2] Initializing Models...")
    models = initialize_models()
    print("    Training models (this might take a moment)...")
    fitted_models = train_models(X_train, y_train, models)

    print("\n[3] Evaluating Models and Generating UI Artifacts...")
    metrics, importances = evaluate_models(fitted_models, X_test, y_test, results_dir)

    print("\n" + "="*50)
    print("Pipeline Execution Complete!")
    print(f"Frontend JSON Artifacts have been successfully saved to:\n  -> {results_dir}")
    print("="*50)

    for m in metrics:
        print(f"[{m['Model']}] R2: {m['R2_Score']:.4f} | RMSE: {m['RMSE']:.2f}")

if __name__ == "__main__":
    main()
