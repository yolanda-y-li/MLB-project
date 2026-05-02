"""
Tiny diagnostic — runs only the bare minimum to expose where things stall.

Usage (from project root):
    cd model
    python tests/diagnose.py

Prints a timestamped log of each step and the wall-clock time it took.
If any step hangs for > a couple of minutes, that's your bottleneck.

Watch in another terminal:
    top -pid $(pgrep -f diagnose.py | head -1)
to see whether Python is actually using CPU. If CPU is near 0% but the
script isn't progressing, the system is swap-thrashing — RAM is the
problem, not compute.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    stamp("Importing torch …")
    t = time.time()
    import torch
    stamp(f"  torch {torch.__version__} imported in {time.time()-t:.1f}s "
          f"(threads: {torch.get_num_threads()})")

    stamp("Importing torch_geometric …")
    t = time.time()
    import torch_geometric  # noqa: F401
    stamp(f"  torch_geometric {torch_geometric.__version__} imported in {time.time()-t:.1f}s")

    stamp("Importing project modules …")
    t = time.time()
    from data_loader import load_data
    from model import GeneDrugRGCN
    stamp(f"  imported in {time.time()-t:.1f}s")

    stamp("Loading data …")
    t = time.time()
    train_graph, splits, info = load_data()
    stamp(f"  loaded in {time.time()-t:.1f}s — "
          f"genes={info['n_genes']:,} drugs={info['n_drugs']:,} "
          f"families={info['n_families']:,} "
          f"train_pos+neg={len(splits['train'][0]):,} "
          f"test_pos+neg={len(splits['test'][0]):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"Device: {device}")
    train_graph = train_graph.to(device)

    stamp("Reading checkpoint args …")
    t = time.time()
    candidates = [
        Path(__file__).parent.parent / "checkpoints" / "best_model.pt",
        Path(__file__).parent.parent / "best_model.pt",
        Path(__file__).parent.parent.parent / "best_model.pt",
    ]
    ckpt_path = next((c for c in candidates if c.exists()), None)
    if ckpt_path is None:
        stamp("  NO CHECKPOINT FOUND — diagnose will skip checkpoint load.")
        ckpt_args = None
    else:
        peek = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = peek.get("args") if isinstance(peek, dict) else None
        stamp(f"  found {ckpt_path.name} ({time.time()-t:.1f}s)  args={ckpt_args}")

    hd = (ckpt_args or {}).get("hidden_dim", 128)
    nl = (ckpt_args or {}).get("num_layers", 3)
    dr = (ckpt_args or {}).get("dropout", 0.05)
    stamp(f"Building model hidden_dim={hd} num_layers={nl} dropout={dr} …")
    t = time.time()
    model = GeneDrugRGCN(
        n_genes=info["n_genes"], n_drugs=info["n_drugs"],
        n_classes=info["n_classes"], n_families=info["n_families"],
        hidden_dim=hd, num_layers=nl, dropout=dr,
        metadata=train_graph.metadata(),
    ).to(device)
    stamp(f"  built in {time.time()-t:.1f}s "
          f"(params: {sum(p.numel() for p in model.parameters()):,})")

    stamp("First full-graph encode (lazy SAGEConv init — usually the slowest step) …")
    t = time.time()
    with torch.no_grad():
        x = model.encode(train_graph.edge_index_dict)
    stamp(f"  first encode finished in {time.time()-t:.1f}s")

    if ckpt_path is not None:
        stamp("Loading checkpoint state …")
        t = time.time()
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        stamp(f"  loaded in {time.time()-t:.1f}s")

    stamp("Second encode (should be much faster than the first) …")
    t = time.time()
    with torch.no_grad():
        x = model.encode(train_graph.edge_index_dict)
    stamp(f"  second encode in {time.time()-t:.1f}s")

    stamp("Predicting on the test split (cached x_dict) …")
    pairs, labels = splits["test"]
    pairs = pairs.to(device)
    t = time.time()
    with torch.no_grad():
        logits = model.classifier(torch.cat(
            [x["compound"][pairs[:, 0]], x["gene"][pairs[:, 1]]], dim=-1))
        probs = torch.softmax(logits, dim=-1)
    stamp(f"  test predictions ({len(pairs):,} pairs) in {time.time()-t:.1f}s")
    stamp(f"  probs shape: {tuple(probs.shape)}  argmax distribution: "
          f"{[int((probs.argmax(1) == c).sum()) for c in range(4)]}")

    stamp("DONE — if you got here, the pipeline works. Run "
          "tests/run_all_tests.py and report which test stalls (it should now "
          "print timestamps for every encode).")


if __name__ == "__main__":
    main()
