SERVER_LLM_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SERVER_LLM_MODELS_DIR="$SERVER_LLM_DIR/models"

# Feel free to use other models
mkdir -p "$SERVER_LLM_MODELS_DIR/$MODEL_LLM_AUTHOR"
mkdir "$SERVER_LLM_MODELS_DIR/$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" && huggingface-cli download "$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" --local-dir "$SERVER_LLM_MODELS_DIR/$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" --local-dir-use-symlinks False
