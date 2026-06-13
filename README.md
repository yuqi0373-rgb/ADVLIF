# ADVLIF

ADVLIF is a multi-modal image translation and segmentation project based on a ParsiLIF training pipeline.  
This version keeps the ADV 1 segmentation head (`unet_adv`) as the active segmentation head.

## Project Structure

```text
ADVLIF/
├── train.py                  # training entry point
├── test.py                   # inference / testing entry point
├── data/                     # dataset loaders
├── models/
│   ├── ParsiLIF_model.py
│   ├── segheads.py           # ADV 1 segmentation head: unet_adv
│   └── down_up.py            # down/up blocks required by unet_adv
├── options/
├── metrics/
└── util/
```

## Environment

Create or activate a Python environment with PyTorch, torchvision, click, tqdm, numpy, Pillow, dask, and the other project dependencies installed.

Example:

```bash
conda activate deepliif_env
```

This codebase uses imports such as `Model.data`, `Model.models`, and `Model.util`. If your repository folder is not named `Model`, run from the parent directory and create a symlink:

```bash
ln -s ADVLIF Model
```

Then run commands from the parent directory, or make sure the parent directory is in `PYTHONPATH`.

## Dataset Format

The default dataset mode is `aligned`. For aligned training, the data directory should contain combined images under:

```text
<DATA_ROOT>/
└── train/
```

For validation/testing, use the corresponding phase folders expected by your dataset setup, such as:

```text
<DATA_ROOT>/
├── train/
├── val/
└── test/
```

The aligned loader expects each sample to be a horizontally concatenated image containing:

```text
A + modality outputs + segmentation/overlay target
```

The default setting uses:

```text
modalities_no = 4
segmentation head input = 15 channels
```

## Training

Basic training command:

```bash
python train.py \
  --dataroot <DATA_ROOT> \
  --name <EXP_NAME> \
  --model ParsiLIF \
  --seghead unet_adv \
  --gpu-ids 0 \
  --checkpoints-dir ./checkpoints \
  --batch-size 1 \
  --load-size 512 \
  --crop-size 512 \
  --n-epochs 5 \
  --n-epochs-decay 5
```

Example:

```bash
python train.py \
  --dataroot ./datasets/my_dataset \
  --name advlif_unet_adv \
  --model ParsiLIF \
  --seghead unet_adv \
  --gpu-ids 0 \
  --checkpoints-dir ./checkpoints
```

Run on CPU:

```bash
python train.py \
  --dataroot <DATA_ROOT> \
  --name <EXP_NAME> \
  --model ParsiLIF \
  --seghead unet_adv \
  --gpu-ids -1
```

Resume training:

```bash
python train.py \
  --dataroot <DATA_ROOT> \
  --name <EXP_NAME> \
  --model ParsiLIF \
  --seghead unet_adv \
  --checkpoints-dir ./checkpoints \
  --continue-train
```

Useful training options:

```text
--dataroot             dataset path
--name                 experiment name
--checkpoints-dir      checkpoint output directory
--model                model class, default: ParsiLIF
--seghead              segmentation head, use: unet_adv
--gpu-ids              GPU ids, e.g. --gpu-ids 0 or --gpu-ids -1 for CPU
--batch-size           batch size
--load-size            resize size
--crop-size            crop size
--n-epochs             epochs at initial learning rate
--n-epochs-decay       epochs for linear LR decay
--lr-g                 generator learning rate
--lr-d                 discriminator learning rate
--with-val             run validation at epoch end
--debug                use debug mode
```

## Testing / Inference

After training, test with:

```bash
python test.py \
  --dataroot <TEST_DATA_ROOT> \
  --name <EXP_NAME> \
  --checkpoints_dir ./checkpoints \
  --gpu_ids 0 \
  --num_test 100
```

Example:

```bash
python test.py \
  --dataroot ./datasets/my_dataset/test \
  --name advlif_unet_adv \
  --checkpoints_dir ./checkpoints \
  --gpu_ids 0 \
  --num_test 100
```

The test script will:

1. Collect the latest checkpoint files into:

```text
checkpoints/<EXP_NAME>/latest/
```

2. Load training options from:

```text
checkpoints/<EXP_NAME>/latest/train_opt.txt
```

3. Save predictions and an HTML visualization to:

```text
<TEST_DATA_ROOT>_pred_<EXP_NAME>/
```

Run inference on CPU:

```bash
python test.py \
  --dataroot <TEST_DATA_ROOT> \
  --name <EXP_NAME> \
  --checkpoints_dir ./checkpoints \
  --gpu_ids -1
```

## ADV 1 Segmentation Head

The active segmentation head is:

```text
unet_adv
```

It is registered in:

```text
models/segheads.py
```

The head uses:

- adversarial local fusion gate (`ALGate`)
- adversarial residual routing skips (`ARRSkip`)
- base encoder/decoder blocks
- reaction-diffusion downsampling
- Poisson-guided upsampling

Other experimental segmentation heads have been removed from this version to keep the repository focused on ADV 1.

## Outputs

During training, checkpoints and logs are saved under:

```text
checkpoints/<EXP_NAME>/
```

Typical checkpoint files include:

```text
latest_net_G1.pth
latest_net_G2.pth
latest_net_G3.pth
latest_net_G4.pth
latest_net_S.pth
train_opt.txt
```

During testing, visual results are saved under:

```text
<TEST_DATA_ROOT>_pred_<EXP_NAME>/
```

## Notes

- Use `--seghead unet_adv`. Other segmentation heads are not included in this cleaned version.
- The default model is `ParsiLIF`.
- The default number of translated modalities is `4`.
- The segmentation head expects a 15-channel input assembled by the model.
