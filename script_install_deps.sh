# Install dependencies

SERVER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

## Main

pip install -r "$SERVER_DIR/requirements/install_deps.txt"

## ExllamaV2

git submodule update --init --recursive
cd "$SERVER_DIR/exllamav2"
python setup.py install
pip install flash-attn --no-build-isolation

## CodeFormer

ln -s "$SERVER_DIR/models/face_restorer/CodeFormer/facelib" "$SERVER_DIR/facelib"
