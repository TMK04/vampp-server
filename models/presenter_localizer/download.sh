SERVER_PL_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SERVER_PL_MODELS_DIR="$SERVER_PL_DIR/models"

mkdir "$SERVER_PL_MODELS_DIR" && huggingface-cli download beholder-vampp/presenter_localizer --local-dir "$SERVER_PL_MODELS_DIR" --local-dir-use-symlink False
