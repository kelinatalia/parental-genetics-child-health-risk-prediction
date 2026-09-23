import joblib
import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # --- DYNAMIC PATH LOGIC ---
    # Check if we are in a SageMaker container or running locally
    if os.environ.get("SM_CHANNEL_TEST") or os.path.exists("/opt/ml/processing"):
        model_path = "/opt/ml/processing/model/model_parental.joblib"
        test_path = "/opt/ml/processing/test/test.csv"
        output_dir = "/opt/ml/processing/evaluation"
    else:
        # Local paths for manual testing on Notebook instance
        model_path = "/home/ec2-user/SageMaker/parental/model/model_parental.joblib"
        test_path = "/home/ec2-user/SageMaker/parental/test/test.csv"
        output_dir = "/home/ec2-user/SageMaker/parental/eval"

    os.makedirs(output_dir, exist_ok=True)

    # Verify files exist before processing
    if os.path.exists(model_path) and os.path.exists(test_path):
        # 1. Load Model and Data
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)
        
        # 2. Prepare Features and Labels
        X_test = test_df.drop("Predicted_Health_Risk", axis=1)
        y_test = test_df["Predicted_Health_Risk"]

        # 3. Predict and Score
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # 4. Create the Report (Must match the structure expected by JsonGet in Pipeline)
        report_dict = {
            "multiclass_classification_metrics": {
                "accuracy": {
                    "value": acc,
                    "standard_deviation": "NaN"}},}

        # 5. Save Report
        output_file_path = os.path.join(output_dir, "evaluation.json")
        with open(output_file_path, "w") as f:
            json.dump(report_dict, f)
            
        print(f"✅ Evaluation complete. Accuracy: {acc:.4f}")
        print(f"Report saved to: {output_file_path}")
    else:
        print("❌ Error: Missing files for evaluation.")
        if not os.path.exists(model_path): print(f"Missing model at: {model_path}")
        if not os.path.exists(test_path): print(f"Missing test data at: {test_path}")