SERVER_XDENSENET_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SERVER_XDENSENET_MODELS_DIR="$SERVER_XDENSENET_DIR/models"

mkdir "$SERVER_XDENSENET_MODELS_DIR" && huggingface-cli download beholder-vampp/xdensenet --local-dir "$SERVER_XDENSENET_MODELS_DIR" --local-dir-use-symlink False
