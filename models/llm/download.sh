# Feel free to use other models
mkdir -p "$ROOT_DIR/models/$MODEL_LLM_DIR" && huggingface-cli download $MODEL_LLM_DIR --local-dir "$ROOT_DIR/models/$MODEL_LLM_DIR" --local-dir-use-symlinks False
mkdir -p "$ROOT_DIR/models/$MODEL_SD_DIR" && huggingface-cli download $MODEL_SD_DIR --local-dir "$ROOT_DIR/models/$MODEL_SD_DIR" --local-dir-use-symlinks False
