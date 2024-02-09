# VAMPP Server

## Prerequisites

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
```

5. Install dependencies

```sh
ROOT_DIR=/ ./script_install_deps.sh # Replace "/" with this directory
```

## Models used

| Task                       | Model                                                                                                                                                                                |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Crop Posture               | Finetuned [YOLOv8n](https://github.com/ultralytics/ultralytics)                                                                                                                      |
| Enhance cropped posture    | [CodeFormer](https://github.com/sczhou/CodeFormer) + [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                                                                           |
| Speech Emotion Recognition | Finetuned [wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)                                                |
| Text Recognition           | [Whisper](https://github.com/openai/whisper)                                                                                                                                         |
| Text Analysis              | [notux-8x7b-v1](https://huggingface.co/jondurbin/bagel-dpo-7b-v0.4) ([5.0bpw-h6-exl2](https://huggingface.co/LoneStriker/bagel-dpo-7b-v0.4-5.0bpw-h6-exl2))                          |
| [Speculative Decoder][SD]  | [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) ([5.0bpw-h6-exl2](https://huggingface.co/LoneStriker/TinyLlama-1.1B-Chat-v1.0-5.0bpw-h6-exl2)) |

[SD]: https://arxiv.org/abs/2211.17192
