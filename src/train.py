import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


DATA_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_model():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

    print("Training model...")
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nModel Evaluation")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_DIR / "xgboost_fraud_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    print("\nModel and scaler saved successfully.")


if __name__ == "__main__":
    train_model()