# Feel free to use other models
mkdir "$ROOT_DIR/models/$MODEL_TRANSCRIBER_AUTHOR"
mkdir "$ROOT_DIR/models/$MODEL_TRANSCRIBER_AUTHOR/$MODEL_TRANSCRIBER_NAME" && huggingface-cli download "$MODEL_TRANSCRIBER_AUTHOR/$MODEL_TRANSCRIBER_NAME" --local-dir "$ROOT_DIR/models/$MODEL_TRANSCRIBER_AUTHOR/$MODEL_TRANSCRIBER_NAME" --local-dir-use-symlinks False
