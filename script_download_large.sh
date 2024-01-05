# Download Large Files

## Models

cd "$ROOT_DIR"
aws s3 cp s3://vampp/models.7z models.7z
7z x models.7z -omodels/
rm models.7z

cd "$ROOT_DIR/models/llm/models"
mkdir notux-8x7b-v1-5.0bpw-h6-exl2 && huggingface-cli download LoneStriker/notux-8x7b-v1-5.0bpw-h6-exl2 --local-dir notux-8x7b-v1-5.0bpw-h6-exl2 --local-dir-use-symlinks False
mkdir TinyLlama-1.1B-Chat-v1.0-5.0bpw-h6-exl2 && huggingface-cli download LoneStriker/TinyLlama-1.1B-Chat-v1.0-5.0bpw-h6-exl2 --local-dir TinyLlama-1.1B-Chat-v1.0-5.0bpw-h6-exl2 --local-dir-use-symlinks False

### CodeFormer

cd "$ROOT_DIR/models/face_restorer/CodeFormer"
python scripts/download_pretrained_models.py all
