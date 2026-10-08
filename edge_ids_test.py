from edge_ids_run_funcs import *

def test(model: Path, scaler_vals_path: Path, windows_path: Path, threshold_path: Path, test_output: Path) -> dict[str, Any]:
    with threshold_path.open("rb") as source:
        threshold_dict = pickle.load(source)
    errors, latencies = evaluate_windows(load_raw_windows(windows_path), EdgeIDSInferenceEngine(model), load_scaler_vals(scaler_vals_path))
    record = {"windows_evaluated": int(len(errors)), "anomalies_flagged": (errors > threshold_dict["threshold"]),
            "threshold": float(threshold_dict["threshold"]), "latency": latency_summary(latencies)}
    test_output.parent.mkdir(parents=True, exist_ok=True)
    with open(test_output, "wb") as f:
        pickle.dump(record, f)
    return record
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scaler-vals", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    args = parser.parse_args()
    result = test(args.model, args.scaler_vals, args.windows, args.threshold, args.test_output)
    print("===========TEST RESULTS============")
    for key in result:
        print(key + " " + str(result[key]) + "\n")

