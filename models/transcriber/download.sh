# Feel free to use other models
mkdir -p "$ROOT_DIR/models/$MODEL_TRANSCRIBER_DIR" && huggingface-cli download $MODEL_TRANSCRIBER_DIR --local-dir "$ROOT_DIR/models/$MODEL_TRANSCRIBER_DIR" --local-dir-use-symlinks False
