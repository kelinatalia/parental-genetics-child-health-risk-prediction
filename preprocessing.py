import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

if __name__ == "__main__":
    # --- DYNAMIC PATH LOGIC ---
    if os.environ.get("SM_CHANNEL_INPUT") or os.path.exists("/opt/ml/processing"):
        input_dir = "/opt/ml/processing/ingested"
        out_train = "/opt/ml/processing/train"
        out_test = "/opt/ml/processing/test"
    else:
        input_dir = "/home/ec2-user/SageMaker/parental/ingested"
        out_train = "/home/ec2-user/SageMaker/parental/train"
        out_test = "/home/ec2-user/SageMaker/parental/test"

    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    input_file = os.path.join(input_dir, "parental_genetics_child_traits.csv")
    
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        df["Family_Disease_History"] = df["Family_Disease_History"].fillna("Unknown")
        df = df.drop("Family_ID", axis = 1)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        encode_tar = LabelEncoder()
        df["Predicted_Health_Risk"] = encode_tar.fit_transform(df["Predicted_Health_Risk"])
        
        categoric = df.select_dtypes(include = "object").columns
        numeric = df.select_dtypes(exclude = "object").columns

        encode = LabelEncoder()
        scale = StandardScaler()

        categoric = list(categoric)
        categoric.remove("Family_ID")
        categoric.remove("Predicted_Health_Risk")

        for i in categoric:
            train_df[i] = encode.fit_transform(train_df[i])
            test_df[i] = encode.transform(test_df[i])

        for i in numeric:
            train_df[i] = scale.fit_transform(train_df[[i]])
            test_df[i] = scale.transform(test_df[[i]])

        
        train_df.to_csv(os.path.join(out_train, "train.csv"), index=False)
        test_df.to_csv(os.path.join(out_test, "test.csv"), index=False)
        print("✅ Preprocessing complete.")
    else:
        print(f"❌ Error: {input_file} not found!")