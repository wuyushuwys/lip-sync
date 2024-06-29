set -xe

# Example of process HDTF dataset (video w/ audio)
# python video_face_extract_facexlib.py \
#         --input_dir /data/HDTF/ \
#         --output_dir /data/HDTF_h5/ \
#         --num_workers 8 \
#         --face_size 256 \
#         --ext png \
#         --name h5_hdtf --cache

# Example of process CMLR dataset (video audio seperate)
python video_face_extract_facexlib.py \
        --input_dir /data/CMLRdataset_video/ \
        --output_dir /data/CMLR_processed/ \
        --num_workers 8 \
        --face_size 256 \
        --ext png \
        --audio_path /data/CMLRdataset_audio \
        --video_path /data/CMLRdataset_video \
        --name CMLRdataset --cache


