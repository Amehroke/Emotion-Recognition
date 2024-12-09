import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import json
import os

# List of CSV files
csv_files = [
    "notebooks/librosa_balanced.csv",
    "notebooks/librosa_extracted_features.csv",
    "notebooks/yamnet_balanced.csv",
    "notebooks/yamnet_extracted_features.csv",
]

# Hyperparameter grid for GridSearchCV
param_grid = {
    "C": [7, 10, 15, 25],
    "class_weight": [None, "balanced"],
}


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


def train_with_grid_search(X_train, y_train):
    model = SVC(random_state=42, kernel="rbf", gamma="auto")
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return accuracy, report


def save_results(file_name, model, accuracy, report):
    results = {
        "file_name": file_name,
        "model_name": type(model).__name__,
        "model_params": model.get_params(),
        "accuracy": accuracy,
        "classification_report": report,
    }
    result_file = f"results_{file_name.split('/')[-1].split('.')[0]}.json"

    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    else:
        existing_results = []

    existing_results.append(results)

    with open(result_file, "w") as f:
        json.dump(existing_results, f, indent=4)


if __name__ == "__main__":
    print("Training SVM model with Grid Search")
    for file in csv_files:
        print(f"Data: {file}")
        print(f"Number of features is {len(pd.read_csv(file).columns)}")
        print(f"Size of the dataset is {len(pd.read_csv(file))}")
        data = load_data(file)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_with_grid_search(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test)
        save_results(file, model, accuracy, report)
        print("-" * 50)
