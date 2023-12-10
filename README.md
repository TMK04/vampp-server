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

3. Download models

```sh
./script_download_large.sh
```

4. Install dependencies

```sh
pip install -r requirements.txt
```
