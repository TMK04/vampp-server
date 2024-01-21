# Download Large Files

. "$ROOT_DIR/.env"

## Models

ROOT_DIR="$ROOT_DIR/models/presenter_localizer" . "$ROOT_DIR/models/presenter_localizer/download.sh"
ROOT_DIR="$ROOT_DIR/models/ridge" . "$ROOT_DIR/models/ridge/download.sh"
ROOT_DIR="$ROOT_DIR/models/speech_stats" . "$ROOT_DIR/models/speech_stats/download.sh"
ROOT_DIR="$ROOT_DIR/models/xdensenet" . "$ROOT_DIR/models/xdensenet/download.sh"

### Larger Models

ROOT_DIR="$ROOT_DIR/models/face_restorer" . "$ROOT_DIR/models/face_restorer/download.sh"
ROOT_DIR="$ROOT_DIR/models/transcriber" MODEL_LLM_DIR="$MODEL_TRANSCRIBER_AUTHOR/$MODEL_TRANSCRIBER_NAME" . "$ROOT_DIR/models/transcriber/download.sh"
ROOT_DIR="$ROOT_DIR/models/llm" MODEL_LLM_DIR="$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" MODEL_SD_DIR="$MODEL_SD_AUTHOR/$MODEL_SD_NAME" . "$ROOT_DIR/models/llm/download.sh"
