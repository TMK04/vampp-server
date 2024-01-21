# Install dependencies

## Main

pip install -r "$ROOT_DIR/requirements/install_deps.txt"

## CodeFormer

ln -s "$ROOT_DIR/models/face_restorer/CodeFormer/facelib" "$ROOT_DIR/facelib"

## Flash Attention

pip install flash-attn --no-build-isolation
