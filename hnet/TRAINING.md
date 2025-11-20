# Training H-Net on CS182 Data

This guide shows how to train H-Net models from scratch on the CS182 dataset.

## Quick Start

### 1. Basic Training

Train on a dataset with default settings:

```bash
cd /home/vartheta0/Downloads/cs182/hnet

python train.py \
  --data-dir ../data/t-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345 \
  --config-path configs/hnet_small.json \
  --output-dir outputs/my_first_model \
  --batch-size 64 \
  --num-epochs 20 \
  --lr 3e-4
```

### 2. Training with GPU (recommended)

If you have a CUDA-enabled GPU, the script will automatically use it. You can also use bfloat16 for faster training:

```bash
python train.py \
  --data-dir ../data/t-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345 \
  --config-path configs/hnet_small.json \
  --output-dir outputs/my_model_bf16 \
  --batch-size 128 \
  --num-epochs 50 \
  --lr 3e-4 \
  --dtype bfloat16
```

### 3. Training with Learning Rate Multipliers

Use different learning rates for different hierarchical stages:

```bash
python train.py \
  --data-dir ../data/t-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345 \
  --config-path configs/hnet_1stage_L.json \
  --output-dir outputs/my_model_lr_mult \
  --batch-size 32 \
  --num-epochs 50 \
  --lr 3e-4 \
  --lr-multipliers "3.0,1.0"
```

## Available Configurations

- `configs/hnet_small.json` - Small model (~1M params) for quick experiments
- `configs/hnet_1stage_L.json` - Large 1-stage model
- `configs/hnet_2stage_L.json` - Large 2-stage hierarchical model
- `configs/hnet_1stage_XL.json` - Extra large 1-stage model
- `configs/hnet_2stage_XL.json` - Extra large 2-stage hierarchical model

## Command-Line Arguments

### Required
- `--data-dir`: Path to the data directory containing train/ and test/ folders

### Optional
- `--config-path`: Model configuration file (default: `configs/hnet_1stage_L.json`)
- `--output-dir`: Where to save checkpoints (default: `outputs`)
- `--batch-size`: Training batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 3e-4)
- `--weight-decay`: Weight decay for regularization (default: 0.1)
- `--aux-loss-weight`: Weight for auxiliary load balancing loss (default: 0.01)
- `--lr-multipliers`: Comma-separated LR multipliers per stage (e.g., "3.0,1.7,0.9")
- `--dtype`: Model dtype: float32, bfloat16, or float16 (default: float32)
- `--save-every`: Save checkpoint every N epochs (default: 1)
- `--eval-every`: Evaluate every N epochs (default: 1)

## Output Files

After training, the output directory will contain:
- `best_model.pt` - Best model based on validation loss
- `final_model.pt` - Model from the last epoch
- `checkpoint_epoch_N.pt` - Checkpoint from epoch N (includes optimizer state)
- `train_args.json` - Training arguments for reproducibility

## Using the Trained Model

Once you have a trained model, you can use it with the generation script:

```bash
python generate.py \
  --model-path outputs/my_model/best_model.pt \
  --config-path configs/hnet_small.json \
  --max-tokens 100
```

Then type a prompt and the model will generate text!

## Tips

1. **Start small**: Use `hnet_small.json` for quick experiments
2. **Monitor GPU memory**: Reduce batch size if you run out of memory
3. **Use bfloat16**: Faster training with minimal accuracy loss (requires recent GPU)
4. **Learning rate**: 3e-4 is a good starting point; adjust based on loss curves
5. **Auxiliary loss**: The `--aux-loss-weight` controls load balancing in hierarchical models

## Example Training Sessions

### Compression Task (Short Sequences)
```bash
python train.py \
  --data-dir ../data/t-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345 \
  --config-path configs/hnet_small.json \
  --output-dir outputs/compression \
  --batch-size 128 \
  --num-epochs 30
```

### Recall Task (Medium Sequences)
```bash
python train.py \
  --data-dir ../data/t-R_vs-32_sl-128_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-16_ntc-0_s-12345 \
  --config-path configs/hnet_1stage_L.json \
  --output-dir outputs/recall \
  --batch-size 64 \
  --num-epochs 50 \
  --dtype bfloat16
```

### Selective Copying (Long Sequences)
```bash
python train.py \
  --data-dir ../data/t-SC_vs-16_sl-256_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-16_s-12345 \
  --config-path configs/hnet_2stage_L.json \
  --output-dir outputs/selective_copying \
  --batch-size 32 \
  --num-epochs 50 \
  --lr-multipliers "3.0,1.7,0.9" \
  --dtype bfloat16
```
