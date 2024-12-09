import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# List of CSV files
csv_files = [
    "notebooks/librosa_balanced.csv",
    "notebooks/librosa_extracted_features.csv",
    "notebooks/yamnet_balanced.csv",
    "notebooks/yamnet_extracted_features.csv",
]


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data):
    X = data.drop(columns=["Label"])
    y = data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    for file in csv_files:
        print(f"Training SVM model on {file}")
        print(f"number of features is {len(pd.read_csv(file).columns)}")
        print(f"size of the dataset is {len(pd.read_csv(file))}")
        data = load_data(file)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        svm_model = train(X_train, y_train)
        evaluate_model(svm_model, X_test, y_test)
        print("-" * 50)
