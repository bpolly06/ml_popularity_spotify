import os
import pandas as pd
from scipy.stats import spearmanr

def correlation_analysis():
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    features_path = os.path.join(project_root, "data/preprocessed_spotify_data.csv")

    # Load preprocessed data
    df = pd.read_csv(features_path)

    # Creating blank results to hold the results
    results = []

    # Compute Spearman correlation and p-values
    target = "popularity"
    for col in df.columns:
        if col == target:
            continue
        corr, pval = spearmanr(df[target], df[col], nan_policy='omit')
        results.append({
            "feature": col,
            "spearman_corr": corr,
            "p_value": pval
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values(by="spearman_corr", ascending=False)

    # Display summary in terminal
    print("\n Top features by Spearman correlation with popularity:")
    print(results_df.head(10))
    print("\n Lowest (negative) correlations:")
    print(results_df.tail(10))

    # Save to CSV for Tableau
    output_path = os.path.join(project_root, "data/spearman_correlation_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n Spearman correlation results saved to: {output_path}")

if __name__ == "__main__":
    correlation_analysis()