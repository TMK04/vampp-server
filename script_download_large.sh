# Download Large Files

SERVER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. "$SERVER_DIR/.env"

mkdir -p "$MODELS_DIR"

## presenter_localizer

mkdir "$MODELS_DIR/presenter_localizer" && huggingface-cli download beholder-vampp/presenter_localizer --local-dir "$MODELS_DIR/presenter_localizer" --local-dir-use-symlink False

## ridge

mkdir "$MODELS_DIR/ridge" && huggingface-cli download beholder-vampp/ridge --local-dir "$MODELS_DIR/ridge" --local-dir-use-symlink False

## speech_stats

mkdir "$MODELS_DIR/speech_stats"
mkdir "$MODELS_DIR/speech_stats/pretrained" && huggingface-cli download audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim --local-dir "$MODELS_DIR/speech_stats/pretrained" --local-dir-use-symlink False --exclude pytorch_model.bin
mkdir "$MODELS_DIR/speech_stats/models" && huggingface-cli download beholder-vampp/speech_stats --local-dir "$MODELS_DIR/speech_stats/models" --local-dir-use-symlink False

## xdensenet

mkdir "$MODELS_DIR/xdensenet" && huggingface-cli download beholder-vampp/xdensenet --local-dir "$MODELS_DIR/xdensenet" --local-dir-use-symlink False

## face_restorer

. "$SERVER_MODELS_DIR/face_restorer/download.sh"

## llm

mkdir -p "$MODELS_DIR/llm/$MODEL_LLM_AUTHOR"
mkdir "$MODELS_DIR/llm/$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" && huggingface-cli download "$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" --local-dir "$MODELS_DIR/llm/$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" --local-dir-use-symlinks False
