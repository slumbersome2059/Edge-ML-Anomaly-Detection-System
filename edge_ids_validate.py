from edge_ids_run_funcs import *
def validate(model: Path, scaler_vals_path: Path, windows_path: Path, threshold_output: Path) -> dict[str, Any]:
    errors, latencies = evaluate_windows(load_raw_windows(windows_path), EdgeIDSInferenceEngine(model), load_scaler_vals(scaler_vals_path))
    record = {"percentile": 99.0, "threshold": float(np.percentile(errors, 99)),
              "validation_error_mean": float(np.mean(errors)), "validation_error_max": float(np.max(errors)),
              "windows_evaluated": int(len(errors)), "latency": latency_summary(latencies)}
    
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_output, "wb") as f:
        pickle.dump(record, f)
    return record

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scaler-vals", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--threshold-output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.model, args.scaler_vals, args.windows, args.threshold_output)
    print("===========VALIDATION RESULTS============")
    for key in result:
        print(key + " " + str(result[key]) + "\n")

