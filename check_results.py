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

# Plot the results
plt.figure(figsize=(12, 6))
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    plt.plot(
        results_df["Dataset"] + " - " + results_df["Model"],
        results_df[metric],
        label=metric,
        marker="o",
    )

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Dataset - Model")
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.tight_layout()

# Save the plot
plot_path = os.path.join(results_dir, "model_performance_comparison.png")
plt.savefig(plot_path)
print(f"Performance comparison plot saved to {plot_path}")

plt.show()
