# Lip-Sync

--------
## todo

- [x] pre-processing instruction
- [ ] dataset brief
- [ ] testing
--------
## Dependencies

```bash
conda create -n lipsync python=3.10
conda activate lipsync

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
--------
## Datasets

- HDTF
- TalkingHead-1KF
- LRS2
- ...
--------
### Data Pre-Processing
We provide data pre-processing tool as in `video_processor`

1. Download the dataset
2. unzip the dataset
3. use [video_processor/video_face_extract_facexlib.py](video_processor/video_face_extract_facexlib.py) to pre-process the dataset
   1. only support single-node multi-gpu pre-processing
   2. only can save to image(jpg,png)/h5/tar file
#### Note
- an example is provided [here](video_processor/run.sh)
- if you save to h5 file. please use [merge_h5.py](merge_h5.py) to link all h5-file to single file
- it is not recommend to save as tar file if you have too many video.
--------
### Dataset Preparation

Put you dataset in the `data` folder. 

You can either edit the `DATA_ROOT` in each dataset file inside [datasets](datasets) or use symbolic link to create folder as `DATA_ROOT` for each dataset.
```bash
ln -s [your dataset] data/[dataset-folder]

# example
ln -s [YOUR_PATH]/CMLR_processed data/CMLR

ln -s [YOUR_PATH]/ffhq-dataset/images1024x1024 data/FFHQ

ln -s [YOUR_PATH]/imagenet data/imagenet

ln -s [YOUR_PATH]/open-images-dataset data/open_images

ln -s [YOUR_PATH]/CelebAMask-HQ/CelebA-HQ-img data/CelebA-HQ

```
- Note: For celebAHQ dataset, only put face images folder to `data`. Do not include mask data.
--------
## Training

- configuration defined in [`config/[exp].yml`](config)

```bash
# example
bash scripts/train_syncnet.bash
bash scripts/train_lipsync.bash
```

