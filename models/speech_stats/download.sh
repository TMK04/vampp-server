SERVER_SS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SERVER_SS_MODELS_DIR="$SERVER_SS_DIR/models"

mkdir "$SERVER_SS_MODELS_DIR" && huggingface-cli download beholder-vampp/speech_stats --local-dir "$SERVER_SS_MODELS_DIR" --local-dir-use-symlink False
