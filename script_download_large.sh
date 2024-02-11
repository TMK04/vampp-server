# Download Large Files

SERVER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. "$SERVER_DIR/.env"
SERVER_MODELS_DIR="$SERVER_DIR/models"

## Models

. "$SERVER_MODELS_DIR/presenter_localizer/download.sh"
. "$SERVER_MODELS_DIR/ridge/download.sh"
. "$SERVER_MODELS_DIR/speech_stats/download.sh"
. "$SERVER_MODELS_DIR/xdensenet/download.sh"

### Larger Models

. "$SERVER_MODELS_DIR/face_restorer/download.sh"
MODEL_LLM_DIR="$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" . "$SERVER_MODELS_DIR/llm/download.sh"
