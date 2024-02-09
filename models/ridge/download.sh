SERVER_RIDGE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SERVER_RIDGE_MODELS_DIR="$SERVER_RIDGE_DIR/models"

mkdir "$SERVER_RIDGE_MODELS_DIR" && huggingface-cli download beholder-vampp/ridge --local-dir "$SERVER_RIDGE_MODELS_DIR" --local-dir-use-symlink False
