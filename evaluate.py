import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import argparse

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
def evaluate_anomaly_detector(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    fault_tags: list,
    output_results_path: Path 
):
    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("==================================================")
    print("      PHASE 4: ANOMALY DETECTION EVALUATION       ")
    print("==================================================")
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positives (Detected Attacks): {tp} | False Positives: {fp}")
    print(f"True Negatives (Clean Windows):    {tn} | False Negatives: {fn}\n")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4%}\n")

    results = {"Precision": precision, "Recall": recall, "F1-Score": f1, "FPR": fpr, "Confusion Matrix": cm}
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path, "wb") as f:
        pickle.dump(results, f)

    # Breakdown detection rate per fault type
    print("--- Recall Breakdown by Injection Type ---")
    df_eval = pd.DataFrame({"fault": fault_tags, "true": y_true, "pred": y_pred})
    for fault in ["rpm_spike", "speed_offset", "throttle_stuck", "gear"]:
        sub = df_eval[df_eval["fault"] == fault]
        if len(sub) > 0:
            det_rate = (sub["pred"] == 1).mean()
            print(f"{fault:<15}: {det_rate:.2%} detected ({sub['pred'].sum()}/{len(sub)})")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--y-true", type=Path, required=True)
    parser.add_argument("--fault-tags", type=Path, required=True)
    parser.add_argument("--output-results-path", type=Path, required=True)
    args = parser.parse_args()
    with args.test_output.open("rb") as source:
        test_output = pickle.load(source)
    with args.y_true.open("rb") as source:
        y_true = pickle.load(source)
    with args.fault_tags.open("rb") as source:
        fault_tags = pickle.load(source)
    evaluate_anomaly_detector(test_output["anomalies_flagged_array"], y_true, fault_tags, args.output_results_path )
if __name__ == "__main__":
    main()
