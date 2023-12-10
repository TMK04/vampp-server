# VAMPP Server

## Prerequisites

- [AWS CLI](https://aws.amazon.com/cli)
- [7zip](https://www.7-zip.org)

## Installation

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
pip install -r "requirements/setup.txt"
```

4. Download models

```sh
ROOT_DIR=/ ./script_download_large.sh # Replace / w/ this directory
```

5. Install dependencies

```sh
pip install -r "requirements/install_deps.txt"
```

## Models used

| Task                       | Model                                                                                                                                                                                                          |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Crop Posture               | Finetuned [YOLOv8n](https://github.com/ultralytics/ultralytics)                                                                                                                                                |
| Enhance cropped posture    | [CodeFormer](https://github.com/sczhou/CodeFormer) + [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                                                                                                     |
| Speech Emotion Recognition | Finetuned [wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)                                                                          |
| Text Recognition           | [Whisper](https://github.com/openai/whisper)                                                                                                                                                                   |
| Text Analysis              | [deepseek-llm-67b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat) ([GPTQ 4bit-32g-actorder_True](https://huggingface.co/TheBloke/deepseek-llm-67b-chat-GPTQ/tree/gptq-4bit-32g-actorder_True)) |
