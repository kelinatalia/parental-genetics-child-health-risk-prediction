import pandas as pd
import os
from xgboost import XGBClassifier
import joblib

if __name__ == "__main__":
    # --- DYNAMIC PATH LOGIC ---
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/home/ec2-user/SageMaker/parental/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/SageMaker/parental/model")
    
    os.makedirs(model_dir, exist_ok=True)
    train_file = os.path.join(train_dir, "train.csv")
    
    if os.path.exists(train_file):
        df = pd.read_csv(train_file)
        X_train = df.drop("Predicted_Health_Risk", axis=1)
        y_train = df["Predicted_Health_Risk"]

        model = XGBClassifier(n_estimators = 100, max_depth =5, learning_rate = 0.2)
        model.fit(X_train, y_train)
        
        joblib.dump(model, os.path.join(model_dir, 'model_parental.joblib'))
        print("✅ Training complete.")
    else:
        print(f"❌ Error: {train_file} not found!")