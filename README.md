# VAMPP Server

## Prerequisites

- [7zip](https://www.7-zip.org)
- [AWS CLI](https://aws.amazon.com/cli)
- [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

## Installation

Note: Replace ROOT_DIR with the directory of this repository

1. Clone

```sh
git clone "https://github.com/TMK04/vampp-server.git" --recurse-submodules -j8
```

2. Set environment variables

```sh
cp .env.example .env # then edit .env
```

3. Install setup dependencies

```sh
ROOT_DIR=/ ./script_setup.sh # Replace "/" with this directory
```

4. Download models

```sh
ROOT_DIR=/ ./script_download_large.sh # Replace "/" with this directory
cd models/llm/models/
# Feel free to use another model; The dir should match MODEL_LLM_DIR in .env
mkdir SUS-Chat-34B-6.0bpw-h6-exl2/
huggingface-cli download LoneStriker/SUS-Chat-34B-6.0bpw-h6-exl2 --local-dir SUS-Chat-34B-6.0bpw-h6-exl2 --local-dir-use-symlinks False
# dir should match MODEL_LLM2_DIR in .env
mkdir loyal-piano-m7-6.0bpw-h6-exl2/
huggingface-cli download LoneStriker/loyal-piano-m7-6.0bpw-h6-exl2 --local-dir loyal-piano-m7-6.0bpw-h6-exl2 --local-dir-use-symlinks False
```

5. Install dependencies

```sh
ROOT_DIR=/ ./script_install_deps.sh # Replace "/" with this directory
```

## Models used

| Task                       | Model                                                                                                                                                    |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Crop Posture               | Finetuned [YOLOv8n](https://github.com/ultralytics/ultralytics)                                                                                          |
| Enhance cropped posture    | [CodeFormer](https://github.com/sczhou/CodeFormer) + [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                                               |
| Speech Emotion Recognition | Finetuned [wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)                    |
| Text Recognition           | [Whisper](https://github.com/openai/whisper)                                                                                                             |
| Text Analysis              | [SUS-Chat-34B](https://huggingface.co/SUSTech/SUS-Chat-34B) ([6.0bpw-h6-exl2](https://huggingface.co/LoneStriker/SUS-Chat-34B-6.0bpw-h6-exl2))           |
| Text Analysis Support      | [loyal-piano-m7](https://huggingface.co/chargoddard/loyal-piano-m7) ([6.0bpw-h8-exl2](https://huggingface.co/LoneStriker/loyal-piano-m7-6.0bpw-h6-exl2)) |
