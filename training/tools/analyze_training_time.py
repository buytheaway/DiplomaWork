"""
Analyze and estimate training time
"""
import json
from pathlib import Path

def main():
    print("=" * 70)
    print("⏱️  TRAINING TIME ANALYSIS")
    print("=" * 70)
    
    # Config parameters
    print("\n📋 Training Configuration:")
    print("   Batch size: 64")
    print("   Epochs: 20")
    print("   Learning rate: 0.1")
    print("   LR scheduler: StepLR (step_size=10, gamma=0.1)")
    print("   Mixed precision: Enabled")
    print("   Device: cuda")
    print("   Model: ResNet50 (IR50)")
    print("   Loss: ArcFace")
    
    # Dataset info
    train_dir = Path("datasets/digiface1m_small/train")
    persons = list(train_dir.glob("*/"))
    total_images = sum(len(list(p.glob("*.png"))) for p in persons)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Persons: {len(persons)}")
    print(f"   Total training images: {total_images}")
    batches_per_epoch = (total_images + 63) // 64
    print(f"   Batches per epoch: {batches_per_epoch}")
    
    # Time estimates
    print(f"\n⏳ Time Estimates (approximate):")
    
    # Typical timing: 0.5-1.0s per batch with CUDA + mixed precision
    time_per_batch_fast = 0.5  # seconds
    time_per_batch_slow = 1.0  # seconds
    
    time_per_epoch_fast = batches_per_epoch * time_per_batch_fast
    time_per_epoch_slow = batches_per_epoch * time_per_batch_slow
    
    print(f"   Per batch: {time_per_batch_fast}-{time_per_batch_slow}s (with CUDA)")
    print(f"   Per epoch: {time_per_epoch_fast:.0f}-{time_per_epoch_slow:.0f}s ({time_per_epoch_fast/60:.1f}-{time_per_epoch_slow/60:.1f} min)")
    
    # Full training
    total_time_fast = time_per_epoch_fast * 20
    total_time_slow = time_per_epoch_slow * 20
    
    print(f"   Full training (20 epochs):")
    print(f"     Fast: {total_time_fast/60:.1f} min ({total_time_fast/3600:.1f} hours)")
    print(f"     Slow: {total_time_slow/60:.1f} min ({total_time_slow/3600:.1f} hours)")
    
    # Current progress
    metrics_path = Path("training/outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        completed_epochs = len(metrics["epochs"])
        print(f"\n✅ Current Training Progress:")
        print(f"   Completed epochs: {completed_epochs}/{20}")
        print(f"   Remaining epochs: {20 - completed_epochs}")
        print(f"   Estimated time remaining: {(20 - completed_epochs) * time_per_epoch_slow / 60:.1f}-{(20 - completed_epochs) * time_per_epoch_fast / 60:.1f} minutes")
        
        print(f"\n📈 Training Metrics:")
        for m in metrics["epochs"]:
            print(f"   Epoch {m['epoch']}:")
            print(f"      Train Loss: {m['train_loss']:.4f}")
            print(f"      Train Acc: {m['train_acc']:.4f}")
            print(f"      Val Acc: {m['val_acc']:.4f}")
    else:
        print(f"\n❌ No training metrics found yet")
    
    print("\n" + "=" * 70)
    print("💡 Tips to Speed Up Training:")
    print("   • Use GPU with CUDA support (currently set)")
    print("   • Use mixed precision (AMP enabled)")
    print("   • Increase batch size (if GPU memory allows)")
    print("   • Reduce num_workers if I/O is bottleneck")
    print("=" * 70)


if __name__ == "__main__":
    main()
