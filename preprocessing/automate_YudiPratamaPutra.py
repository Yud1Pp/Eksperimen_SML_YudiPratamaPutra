import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import argparse

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded: {df.shape}")
    return df

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing = df['TotalCharges'].isnull().sum()
    print(f"[INFO] Hidden missing TotalCharges: {missing} baris → di-fill 0")

    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['customerID'], inplace=True)
    print(f"[INFO] customerID di-drop. Shape: {df.shape}")
    return df

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map(binary_map)
    print("[INFO] Binary encoding selesai.")
    return df

def encode_ohe(df: pd.DataFrame) -> pd.DataFrame:
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"[INFO] OHE selesai. Shape: {df.shape}")
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("[INFO] Scaling selesai.")
    return df

def handle_imbalance(X: pd.DataFrame, y: pd.Series):
    print(f"[INFO] Sebelum SMOTE: {y.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[INFO] Setelah SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res

def save_outputs(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    print(f"[INFO] Output disimpan ke '{output_dir}/'")

def main(input_path: str, output_dir: str):
    df = load_data(input_path)
    df = fix_total_charges(df)
    df = drop_irrelevant(df)
    df = encode_binary(df)
    df = encode_ohe(df)
    df = scale_features(df)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_res, y_res = handle_imbalance(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    save_outputs(X_train, X_test, y_train, y_test, output_dir)
    print(f"✅ Preprocessing selesai! X_train: {X_train.shape} | X_test: {X_test.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--output', type=str,
                        default='telco_preprocessing')
    args = parser.parse_args()
    main(args.input, args.output)