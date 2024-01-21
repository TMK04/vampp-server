# Feel free to use other models
mkdir "$ROOT_DIR/models/$MODEL_LLM_AUTHOR"
mkdir "$ROOT_DIR/models/$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" && huggingface-cli download "$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" --local-dir "$ROOT_DIR/models/$MODEL_LLM_AUTHOR/$MODEL_LLM_NAME" --local-dir-use-symlinks False
mkdir "$ROOT_DIR/models/$MODEL_SD_AUTHOR"
mkdir "$ROOT_DIR/models/$MODEL_SD_AUTHOR/$MODEL_SD_NAME" && huggingface-cli download "$MODEL_SD_AUTHOR/$MODEL_SD_NAME" --local-dir "$ROOT_DIR/models/$MODEL_SD_AUTHOR/$MODEL_SD_NAME" --local-dir-use-symlinks False
