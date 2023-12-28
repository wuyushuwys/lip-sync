# Lip-Sync

## todo

- [ ] pre-processing instruction
- [ ] dataset brief
- [ ] testing

## Dependencies

```bash
conda create -n lipsync python=3.10
conda activate lipsync

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Datasets

- HDTF
- TalkingHead-1KF

### Data pre-processing

## Training

- configuration defined in `config/[exp].yml`

```bash
bash scripts/train_syncnet.bash
bash scripts/train_lipsync.bash
```

