import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Specify the directory containing the result JSON files
results_dir = "./logs"

# List to store all results
all_results = []

# Loop through all JSON files in the directory
for file_name in os.listdir(results_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, "r") as f:
            data = json.load(f)
            # Assuming data is a list of dictionaries
            all_results.extend(data)

# Convert all results into a DataFrame
results_df = pd.DataFrame([
    {
        "Dataset": os.path.basename(result["file_name"]).replace(".csv", ""),
        "Model": result["model_name"],
        "Accuracy": result["accuracy"],
        "Precision": result["classification_report"]["weighted avg"]["precision"],
        "Recall": result["classification_report"]["weighted avg"]["recall"],
        "F1-Score": result["classification_report"]["weighted avg"]["f1-score"],
    }
    for result in all_results
])

# Save results to a CSV file
results_csv_path = os.path.join(results_dir, "results_summary.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results summary saved to {results_csv_path}")

# Create datasets and models for plotting
datasets_models = results_df["Dataset"] + " - " + results_df["Model"]

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(
    datasets_models, results_df["Accuracy"], label="Accuracy", marker="o", color="blue"
)
plt.title("Model Performance: Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Dataset - Model")
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.tight_layout()
plot_accuracy_path = os.path.join(results_dir, "model_performance_accuracy.png")
plt.savefig(plot_accuracy_path)
print(f"Accuracy plot saved to {plot_accuracy_path}")
plt.show()

# Plot Precision, Recall, and F1-Score
plt.figure(figsize=(10, 6))
for metric in ["Precision", "Recall", "F1-Score"]:
    plt.plot(datasets_models, results_df[metric], label=metric, marker="o")
plt.title("Model Performance: Precision, Recall, and F1-Score")
plt.ylabel("Score")
plt.xlabel("Dataset - Model")
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.tight_layout()
plot_other_metrics_path = os.path.join(
    results_dir, "model_performance_other_metrics.png"
)
plt.savefig(plot_other_metrics_path)
print(f"Other metrics plot saved to {plot_other_metrics_path}")
plt.show()
