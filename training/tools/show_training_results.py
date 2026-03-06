"""
Show training results
"""
import json
from pathlib import Path

metrics_path = Path("training/outputs/metrics.json")
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    print("=" * 70)
    print("✅ TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nTotal epochs: {len(metrics['epochs'])}")
    print("\n📊 Training Metrics:")
    print("Epoch | Train Loss | Train Acc | Val Acc")
    print("-" * 45)
    
    for m in metrics["epochs"]:
        epoch = m["epoch"]
        loss = m["train_loss"]
        train_acc = m["train_acc"]
        val_acc = m["val_acc"]
        print(f"{epoch:5d} | {loss:10.4f} | {train_acc:9.4f} | {val_acc:7.4f}")
    
    print("\n📁 Checkpoints created:")
    output_dir = Path("training/outputs")
    for ckpt in sorted(output_dir.glob("checkpoint_epoch_*.pth")):
        size_mb = ckpt.stat().st_size / (1024*1024)
        print(f"   {ckpt.name}: {size_mb:.1f} MB")
    
    print("\n" + "=" * 70)
else:
    print("No metrics found")
