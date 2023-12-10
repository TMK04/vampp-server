# Download Large Files

## Models

cd "$ROOT_DIR"
aws s3 cp s3://vampp/models.7z models.7z
7z x models.7z -omodels/
rm models.7z

### CodeFormer

cd "$ROOT_DIR/CodeFormer"
python basicsr/setup.py develop
python scripts/download_pretrained_models.py CodeFormer
