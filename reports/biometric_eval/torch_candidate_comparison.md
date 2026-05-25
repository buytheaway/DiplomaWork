# Torch Candidate Comparison

Technical candidate comparison for custom Torch embedding checkpoints. This report does not change backend/runtime checkpoints and does not rebuild FAISS indexes.

| Model / checkpoint | Architecture | TTA | EER | Best accuracy | TAR@FAR=0.01 | TAR@FAR=0.001 | Threshold | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| reference_baseline (`models/w600k_r50.onnx`) | onnx_w600k_r50 | none | 0.027852 | 0.984556 | 0.971141 | 0.969128 | 0.252612 | external pretrained ONNX/InsightFace reference baseline |
| distill_C_hflip (`training/runs/custom_distill_arcface/C_lambda1.0_lr5e-6/best_by_eer.pth`) | ir50 | hflip | 0.136667 | 0.868333 | 0.623333 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| distill_B_hflip (`training/runs/custom_distill_arcface/B_lambda0.5_lr1e-5/best_by_eer.pth`) | ir50 | hflip | 0.136667 | 0.868333 | 0.626667 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_lfw_ep005_hflip (`diplomcheckbackup/training/outputs_medium_lfw_finetune/checkpoint_epoch_005.pth`) | ir50 | hflip | 0.143333 | 0.865000 | 0.586667 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| handoff_best_hflip (`handoff_lfw_eval/artifacts/best_lfw.pth`) | ir50 | hflip | 0.146667 | 0.863333 | 0.583333 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_lfw_ep005_pretrained_align (`diplomcheckbackup/training/outputs_medium_lfw_finetune/checkpoint_epoch_005.pth`) | ir50 | none | 0.150000 | 0.853088 | 0.520000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| best distill custom (`training/runs/custom_distill_arcface/A_lambda0.25_lr1e-5/checkpoint_epoch_004.pth`) | ir50 | hflip | 0.151333 | 0.851000 | 0.492667 | 0.232333 | 0.359338 | best previous full distillation result |
| distill_B_best (`training/runs/custom_distill_arcface/B_lambda0.5_lr1e-5/best_by_eer.pth`) | ir50 | none | 0.156667 | 0.865000 | 0.463333 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| distill_C_best (`training/runs/custom_distill_arcface/C_lambda1.0_lr5e-6/best_by_eer.pth`) | ir50 | none | 0.156667 | 0.863333 | 0.440000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| distill_B_pretrained_align (`training/runs/custom_distill_arcface/B_lambda0.5_lr1e-5/best_by_eer.pth`) | ir50 | none | 0.156667 | 0.848080 | 0.513333 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| handoff_best_lfw (`handoff_lfw_eval/artifacts/best_lfw.pth`) | ir50 | none | 0.160000 | 0.851667 | 0.490000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| aligned_distill_A_best (`training/runs/aligned_distill_full_A3/A_lambda0.25_lr1e-5/best_by_eer.pth`) | ir50 | none | 0.160000 | 0.851667 | 0.500000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_lfw_ep003 (`diplomcheckbackup/training/outputs_medium_lfw_finetune/checkpoint_epoch_003.pth`) | ir50 | none | 0.160000 | 0.851667 | 0.490000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_best_lfw (`diplomcheckbackup/training/outputs_medium_lfw_finetune/best_lfw.pth`) | ir50 | none | 0.160000 | 0.851667 | 0.490000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_lfw_ep005 (`diplomcheckbackup/training/outputs_medium_lfw_finetune/checkpoint_epoch_005.pth`) | ir50 | none | 0.163333 | 0.856667 | 0.480000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_lfw_ep012 (`diplomcheckbackup/training/outputs_medium_lfw_finetune/checkpoint_epoch_012.pth`) | ir50 | none | 0.166667 | 0.850000 | 0.476667 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_adaface_ep030 (`diplomcheckbackup/training/outputs_medium_adaface/checkpoint_epoch_030.pth`) | ir50 | none | 0.166667 | 0.843333 | 0.506667 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| old custom default (`handoff_lfw_eval/artifacts/best_lfw.pth`) | ir50 | none | 0.166949 | 0.834350 | 0.496449 | 0.213730 | 0.322468 | current old custom LFW full result |
| backup_realface_low_lr_ep007 (`diplomcheckbackup/training/outputs_realface_low_lr/checkpoint_epoch_007.pth`) | ir50 | none | 0.170000 | 0.841667 | 0.513333 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_warmstart_ep080 (`diplomcheckbackup/training/outputs_medium_warmstart/checkpoint_epoch_080.pth`) | ir50 | none | 0.183333 | 0.830000 | 0.493333 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_realface_warmstart_ep040 (`diplomcheckbackup/training/outputs_realface_warmstart/checkpoint_epoch_040.pth`) | ir50 | none | 0.203333 | 0.803333 | 0.413333 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_medium36h_ep150 (`diplomcheckbackup/training/outputs_medium36h/checkpoint_epoch_150.pth`) | ir50 | none | 0.284392 | 0.716667 | 0.000000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_digiface_ep032 (`diplomcheckbackup/training/outputs_digiface_7h/checkpoint_epoch_032.pth`) | ir50 | none | 0.290000 | 0.711667 | 0.080000 |  |  | 600-pair smoke/variant evaluation; not full LFW |
| backup_outputs_ep012 (`diplomcheckbackup/training/outputs/checkpoint_epoch_012.pth`) | ir50 | none | 0.450000 | 0.568333 | 0.003333 |  |  | 600-pair smoke/variant evaluation; not full LFW |

## Interpretation

- No evaluated Torch checkpoint reached the strong-candidate threshold: accuracy >= 0.95, EER <= 0.07, or TAR@FAR=0.01 >= 0.80.
- The best observed Torch candidate remains far below the pretrained ONNX/InsightFace reference baseline.
- Runtime replacement is not recommended based on these metrics alone.
