SERVER_SS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir "$SERVER_SS_DIR/pretrained" && huggingface-cli download audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim --local-dir "$SERVER_SS_DIR/pretrained" --local-dir-use-symlink False --exclude pytorch_model.bin
mkdir "$SERVER_SS_DIR/models" && huggingface-cli download beholder-vampp/speech_stats --local-dir "$SERVER_SS_DIR/models" --local-dir-use-symlink False
