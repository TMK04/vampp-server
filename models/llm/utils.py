import torch

from .tokenizer import tokenizer

col_default = "\u001b[0m"
col_user = "\u001b[33;1m"  # Yellow
col_bot = "\u001b[34;1m"  # Blue
col_error = "\u001b[31;1m"  # Magenta
col_sysprompt = "\u001b[37;1m"  # Grey

empty_ids = torch.tensor([], dtype=torch.long)


def tokenizePrompt(prompt: str):
  printPrompt(prompt)
  return tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)


def printChunks(*args, **kwargs):
  print(*args, end="", **kwargs)


def printSysPrompt(*args, **kwargs):
  print(f"{col_sysprompt}===", *args, end=" ===\n", **kwargs)


def printEOS(reason: str):
  print(end=" ")
  printSysPrompt(f"EOS: {reason}")


def printPrompt(prompt: str):
  printSysPrompt("Prompt")
  printChunks(col_default + prompt)
