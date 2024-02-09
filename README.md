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
- LRS2
- ...
### Dataset Preparation

Put you dataset in the `data` folder. 

You can either edit the `DATA_ROOT` in each dataset file inside [datasets](datasets) or use symbolic link to create folder as `DATA_ROOT` for each dataset.
```bash
ln -s [your dataset] data/[dataset-folder]

# example
ln -s [your path to FFHQ] data/FFHQ
```
- Note: For celebAHQ dataset, only put images folder to `data`. Do not include mask data.

### Data pre-processing

## Training

- configuration defined in [`config/[exp].yml`](config)

```bash
bash scripts/train_syncnet.bash
bash scripts/train_lipsync.bash
```

