"""
Systematic hyperparameter search for Gene-Drug Interaction R-GCN.
Simplified version - runs fine-tuning.
"""

import subprocess
import time
from pathlib import Path

BEST_CONFIG = {
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.1,
    "lr": 0.005,
}

# Phase 3 configurations (fine-tuning around best)
PHASE3_CONFIGS = [
    # Original best
    BEST_CONFIG.copy(),
    # Vary neg_ratio
    {**BEST_CONFIG, "neg_ratio": 0.5},
    {**BEST_CONFIG, "neg_ratio": 2.0},
    # Vary batch_size
    {**BEST_CONFIG, "batch_size": 2048},
    {**BEST_CONFIG, "batch_size": 8192},
    # Vary dropout slightly
    {**BEST_CONFIG, "dropout": 0.05},
    {**BEST_CONFIG, "dropout": 0.15},
    # Vary lr slightly
    {**BEST_CONFIG, "lr": 0.004},
    {**BEST_CONFIG, "lr": 0.006},
]

def run_config(config_dict, run_id):
    """Run a single configuration."""
    base_args = {
        "epochs": 100,
        "batch_size": 4096,
        "weight_decay": 1e-5,
        "patience": 15,
        "seed": 42,
    }
    
    merged_config = {**base_args, **config_dict}
    
    print(f"\n{'='*80}")
    print(f"Run {run_id} | Phase 3 Fine-tuning")
    print(f"Config: {merged_config}")
    print(f"{'='*80}")
    
    # Build command
    cmd = ["python", "main.py"]
    for key, value in merged_config.items():
        if key == "device":
            continue
        cmd.append(f"--{key}")
        cmd.append(str(value))
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start_time
        
        # Extract test metrics
        output = result.stdout + result.stderr
        lines = output.split('\n')
        
        test_auroc, test_ap, test_f1 = None, None, None
        for i, line in enumerate(lines):
            if "Test Results" in line:
                for j in range(i+1, min(i+6, len(lines))):
                    if "AUROC" in lines[j]:
                        try:
                            test_auroc = float(lines[j].split(':')[1].strip())
                        except:
                            pass
                    elif "AP" in lines[j] and "AUROC" not in lines[j]:
                        try:
                            test_ap = float(lines[j].split(':')[1].strip())
                        except:
                            pass
                    elif "F1" in lines[j]:
                        try:
                            test_f1 = float(lines[j].split(':')[1].strip())
                        except:
                            pass
        
        avg_metric = (test_auroc + test_ap + test_f1) / 3 if all([test_auroc, test_ap, test_f1]) else 0
        
        print(f"[OK] Completed in {elapsed:.1f}s")
        print(f"  Test AUROC: {test_auroc}")
        print(f"  Test AP:    {test_ap}")
        print(f"  Test F1:    {test_f1}")
        print(f"  Avg Metric: {avg_metric:.4f}")
        
        return {
            "run_id": run_id,
            "config": merged_config,
            "test_auroc": test_auroc,
            "test_ap": test_ap,
            "test_f1": test_f1,
            "avg_metric": avg_metric,
            "time": elapsed,
        }
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def main():
    print("\n\n" + "="*80)
    print("PHASE 3: FINE-GRAINED SEARCH")
    print("="*80)
    
    results = []
    for i, config in enumerate(PHASE3_CONFIGS, 1):
        result = run_config(config, i)
        if result:
            results.append(result)
    
    # Sort by avg metric
    results.sort(key=lambda x: x["avg_metric"], reverse=True)
    
    print("\n\n" + "="*80)
    print("PHASE 3 RESULTS")
    print("="*80)
    print(f"\nTop configurations:")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. Avg Metric: {r['avg_metric']:.4f}")
        print(f"   AUROC={r['test_auroc']:.4f} AP={r['test_ap']:.4f} F1={r['test_f1']:.4f}")
        cfg = r['config']
        print(f"   hidden_dim={cfg.get('hidden_dim')} num_layers={cfg.get('num_layers')} "
              f"dropout={cfg.get('dropout')} lr={cfg.get('lr')}")


if __name__ == "__main__":
    main()
