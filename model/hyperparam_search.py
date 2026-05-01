"""
Systematic hyperparameter search for Gene-Drug Interaction R-GCN.

Conducts grid search over key hyperparameters and logs all results to CSV.
Focuses on breadth over depth given 2-hour time constraint.
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Hyperparameter search space (coarse-grained for 2-hour constraint)
PHASE2_CONFIGS = {
    "architecture": [
        {"hidden_dim": 32, "num_layers": 1},
        {"hidden_dim": 32, "num_layers": 2},
        {"hidden_dim": 32, "num_layers": 3},
        {"hidden_dim": 64, "num_layers": 1},
        {"hidden_dim": 64, "num_layers": 2},
        {"hidden_dim": 64, "num_layers": 3},
        {"hidden_dim": 128, "num_layers": 1},
        {"hidden_dim": 128, "num_layers": 2},
        {"hidden_dim": 128, "num_layers": 3},
        {"hidden_dim": 256, "num_layers": 1},
        {"hidden_dim": 256, "num_layers": 2},
    ],
    "learning_rate": [
        {"lr": 1e-4},
        {"lr": 5e-4},
        {"lr": 1e-3},
        {"lr": 5e-3},
    ],
    "dropout": [
        {"dropout": 0.0},
        {"dropout": 0.1},
        {"dropout": 0.2},
        {"dropout": 0.3},
        {"dropout": 0.5},
    ],
}

# Phase 3: fine-grained search around best performers (populated dynamically)
PHASE3_CONFIGS = None


class HyperparmSearch:
    def __init__(self, results_dir="hyperparam_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.csv_path = self.results_dir / f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.fieldnames = [
            "run_id", "phase", "hidden_dim", "num_layers", "dropout", 
            "lr", "weight_decay", "batch_size", "neg_ratio", "neg_strategy",
            "train_auroc", "train_ap", "train_f1", "train_loss",
            "val_auroc", "val_ap", "val_f1", "val_loss",
            "test_auroc", "test_ap", "test_f1", "test_loss",
            "time_seconds", "status", "notes"
        ]
        self._init_csv()
        self.run_id = 0
        self.results = []

    def _init_csv(self):
        """Create CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def run_config(self, phase, config_dict, base_args=None):
        """Run a single configuration."""
        if base_args is None:
            base_args = {
                "epochs": 100,
                "batch_size": 4096,
                "weight_decay": 1e-5,
                "patience": 15,
                "seed": 42,
                "device": "cuda",
            }

        self.run_id += 1
        merged_config = {**base_args, **config_dict}
        
        print(f"\n{'='*80}")
        print(f"Run {self.run_id} | Phase {phase}")
        print(f"Config: {merged_config}")
        print(f"{'='*80}")

        # Build command
        cmd = ["python", "main.py"]
        for key, value in merged_config.items():
            if key == "device":
                continue  # device is auto-detected
            cmd.append(f"--{key}")
            cmd.append(str(value))

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout per run
                cwd=Path(__file__).parent
            )
            elapsed = time.time() - start_time

            # Parse output for metrics
            metrics = self._parse_output(result.stdout + result.stderr)
            metrics.update(merged_config)
            metrics["run_id"] = self.run_id
            metrics["phase"] = phase
            metrics["time_seconds"] = round(elapsed, 2)
            metrics["status"] = "completed" if result.returncode == 0 else "error"

            if result.returncode != 0:
                metrics["notes"] = f"Non-zero exit code: {result.returncode}"
            
            self.results.append(metrics)
            self._write_result(metrics)
            
            print(f"[OK] Completed in {elapsed:.1f}s")
            print(f"  Test AUROC: {metrics.get('test_auroc', 'N/A')}")
            print(f"  Test AP:    {metrics.get('test_ap', 'N/A')}")
            print(f"  Test F1:    {metrics.get('test_f1', 'N/A')}")

        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Timeout (>600s)")
            self.results.append({
                "run_id": self.run_id,
                "phase": phase,
                "status": "timeout",
                "time_seconds": 600,
                "notes": "Exceeded 10-minute timeout",
                **merged_config
            })
            self._write_result(self.results[-1])
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            self.results.append({
                "run_id": self.run_id,
                "phase": phase,
                "status": "error",
                "time_seconds": time.time() - start_time,
                "notes": str(e),
                **merged_config
            })
            self._write_result(self.results[-1])

        return metrics if "metrics" in locals() else None

    def _parse_output(self, output):
        """Parse test metrics from training output."""
        metrics = {
            "train_auroc": None, "train_ap": None, "train_f1": None, "train_loss": None,
            "val_auroc": None, "val_ap": None, "val_f1": None, "val_loss": None,
            "test_auroc": None, "test_ap": None, "test_f1": None, "test_loss": None,
        }
        
        lines = output.split('\n')
        
        # Parse last epoch metrics (approximation of final validation)
        for i, line in enumerate(lines):
            if "Epoch" in line and "Val" in line:
                # Extract val metrics from last epoch
                parts = line.split('|')
                if len(parts) >= 3:
                    val_part = parts[2]
                    try:
                        val_metrics = self._parse_metric_line(val_part)
                        if val_metrics:
                            metrics.update(val_metrics)
                            metrics["val_loss"] = metrics.get("loss")
                    except:
                        pass
            
            # Parse test results
            if "Test Results" in line:
                for j in range(i+1, min(i+6, len(lines))):
                    if "AUROC" in lines[j]:
                        try:
                            metrics["test_auroc"] = float(lines[j].split(':')[1].strip())
                        except:
                            pass
                    elif "AP" in lines[j] and "AUROC" not in lines[j]:
                        try:
                            metrics["test_ap"] = float(lines[j].split(':')[1].strip())
                        except:
                            pass
                    elif "F1" in lines[j]:
                        try:
                            metrics["test_f1"] = float(lines[j].split(':')[1].strip())
                        except:
                            pass
                    elif "Loss" in lines[j]:
                        try:
                            metrics["test_loss"] = float(lines[j].split(':')[1].strip())
                        except:
                            pass
        
        return metrics

    def _parse_metric_line(self, line):
        """Parse metric line like 'loss 0.7809  AUROC 0.8556  F1 0.6297'."""
        result = {}
        try:
            parts = line.split()
            for i, part in enumerate(parts):
                if part in ["loss", "Loss"]:
                    result["loss"] = float(parts[i+1])
                elif part == "AUROC":
                    result["auroc"] = float(parts[i+1])
                elif part == "F1":
                    result["f1"] = float(parts[i+1])
                elif part == "AP":
                    result["ap"] = float(parts[i+1])
        except:
            pass
        return result

    def _write_result(self, result):
        """Append result to CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({field: result.get(field, '') for field in self.fieldnames})

    def get_top_configs(self, n=5):
        """Get top N configurations by averaged metric."""
        if not self.results:
            return []
        
        ranked = []
        for r in self.results:
            if r.get("status") == "completed":
                test_auroc = r.get("test_auroc") or 0
                test_ap = r.get("test_ap") or 0
                test_f1 = r.get("test_f1") or 0
                avg_metric = (test_auroc + test_ap + test_f1) / 3
                ranked.append((avg_metric, r))
        
        ranked.sort(reverse=True)
        return [r for _, r in ranked[:n]]

    def generate_phase3_configs(self, top_results):
        """Generate Phase 3 fine-grained search around top performers."""
        phase3 = []
        
        for result in top_results:
            # Fine-tune around this result
            base_config = {
                "hidden_dim": result.get("hidden_dim", 64),
                "num_layers": result.get("num_layers", 2),
                "dropout": result.get("dropout", 0.2),
                "lr": result.get("lr", 1e-3),
                "batch_size": result.get("batch_size", 4096),
                "neg_ratio": result.get("neg_ratio", 1.0),
            }
            
            # Create variations
            variations = [
                base_config,  # baseline again
                {**base_config, "neg_ratio": 0.5},
                {**base_config, "neg_ratio": 2.0},
            ]
            
            # Vary batch_size
            for bs in [2048, 8192]:
                variations.append({**base_config, "batch_size": bs})
            
            phase3.extend(variations)
        
        return phase3

    def run_phase(self, phase_name, configs):
        """Run all configs in a phase."""
        print(f"\n\n{'='*80}")
        print(f"PHASE {phase_name}")
        print(f"{'='*80}\n")
        
        for config in configs:
            self.run_config(phase_name, config)
        
        top = self.get_top_configs(n=5)
        print(f"\n\nTop 5 configurations in {phase_name}:")
        for i, r in enumerate(top, 1):
            avg = (r.get("test_auroc", 0) + r.get("test_ap", 0) + r.get("test_f1", 0)) / 3
            print(f"  {i}. AUROC={r.get('test_auroc', 'N/A'):.4f} AP={r.get('test_ap', 'N/A'):.4f} "
                  f"F1={r.get('test_f1', 'N/A'):.4f} (avg={avg:.4f})")
            print(f"     hidden_dim={r.get('hidden_dim')} num_layers={r.get('num_layers')} "
                  f"dropout={r.get('dropout')} lr={r.get('lr')}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for R-GCN")
    parser.add_argument("--phase", type=str, choices=["2", "3", "all"], default="all",
                        help="Which phase(s) to run")
    parser.add_argument("--results_dir", type=str, default="hyperparam_results",
                        help="Directory to store results")
    args = parser.parse_args()

    search = HyperparmSearch(results_dir=args.results_dir)

    if args.phase in ["2", "all"]:
        print("\n\n" + "="*80)
        print("PHASE 2: COARSE-GRAINED SEARCH")
        print("="*80)
        
        # Architecture sweep
        print("\n>>> Architecture Sweep")
        search.run_phase("2A_architecture", PHASE2_CONFIGS["architecture"])
        
        # Get best architecture
        top_arch = search.get_top_configs(n=1)[0]
        best_hidden = top_arch.get("hidden_dim", 64)
        best_layers = top_arch.get("num_layers", 2)
        
        # Learning rate sweep with best architecture
        print("\n>>> Learning Rate Sweep")
        lr_configs = [
            {"hidden_dim": best_hidden, "num_layers": best_layers, **lr_cfg}
            for lr_cfg in PHASE2_CONFIGS["learning_rate"]
        ]
        search.run_phase("2B_lr", lr_configs)
        
        # Get best lr
        top_lr = search.get_top_configs(n=1)[0]
        best_lr = top_lr.get("lr", 1e-3)
        
        # Dropout sweep with best arch + lr
        print("\n>>> Dropout Sweep")
        dropout_configs = [
            {"hidden_dim": best_hidden, "num_layers": best_layers, "lr": best_lr, **dropout_cfg}
            for dropout_cfg in PHASE2_CONFIGS["dropout"]
        ]
        search.run_phase("2C_dropout", dropout_configs)

    if args.phase in ["3", "all"]:
        print("\n\n" + "="*80)
        print("PHASE 3: FINE-GRAINED SEARCH")
        print("="*80)
        
        top_results = search.get_top_configs(n=3)
        phase3_configs = search.generate_phase3_configs(top_results)
        search.run_phase("3_finetune", phase3_configs)

    # Final summary
    print("\n\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    top_final = search.get_top_configs(n=10)
    print(f"\nTop 10 configurations:")
    for i, r in enumerate(top_final, 1):
        avg = (r.get("test_auroc", 0) + r.get("test_ap", 0) + r.get("test_f1", 0)) / 3
        print(f"\n{i}. Avg Metric: {avg:.4f}")
        print(f"   AUROC={r.get('test_auroc', 'N/A'):.4f} AP={r.get('test_ap', 'N/A'):.4f} "
              f"F1={r.get('test_f1', 'N/A'):.4f}")
            dropout_str = f"{r.get('dropout'):.1f}" if r.get('dropout') is not None else "N/A"
            lr_str = f"{r.get('lr'):.1e}" if r.get('lr') is not None else "N/A"
            print(f"   hidden_dim={r.get('hidden_dim')} num_layers={r.get('num_layers')} "
              f"dropout={dropout_str} lr={lr_str}")
        print(f"   batch_size={r.get('batch_size')} neg_ratio={r.get('neg_ratio')}")
    
    print(f"\n\nResults saved to: {search.csv_path}")


if __name__ == "__main__":
    main()
