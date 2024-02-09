SERVER_FR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SERVER_FR_DIR/CodeFormer"
python scripts/download_pretrained_models.py all
