import sys
import torch
from exllamav2.generator import ExLlamaV2Sampler

from server.models.llm.generator import generator
from server.models.llm.tokenizer import tokenizer
from server.models.llm.utils import col_bot, col_default, empty_ids, printChunks, printEOS

from .Cogenerator import Cogenerator

settings = ExLlamaV2Sampler.Settings()
settings.token_repetition_penalty = 1.1
settings.top_k = 1


def cogenerateSingle(
    cogenerator: Cogenerator,
    input_ids: torch.Tensor = empty_ids,
    max_new_tokens: int = 64,
):
  # Send prompt to generator to begin stream

  generator.begin_stream(input_ids, settings)

  # Streaming loop. Note that repeated calls to sys.stdout.flush() adds some latency, but some
  # consoles won't update partial lines without it.

  generated_tokens = 0

  printChunks(col_bot)
  chunk_prev = ""
  while True:
    chunk, eos, chunk_tokens = generator.stream()
    if eos:
      printEOS("exllamav2")
      cogenerator.setCurrentV()
      break

    printChunks(chunk)
    sys.stdout.flush()
    chunk_prev += chunk
    yield chunk_prev

    for token_tensor in chunk_tokens[0]:
      generated_tokens += 1
      token = token_tensor.item()
      # ? handle comments
      match token:
        case tokenizer.newline_token_id:
          if len(cogenerator.current_v) > 0:
            printEOS("matched newline_token_id")
            done, next_tokens = cogenerator.setCurrentV()
            if done:
              printEOS("done")
              eos = True
              break
            if next_tokens is None:
              input_ids = generator.sequence_ids
            else:
              printChunks(col_default,
                          tokenizer.decode(next_tokens, decode_special_tokens=True),
                          col_bot,
                          sep="")
              input_ids = torch.cat((generator.sequence_ids, next_tokens.unsqueeze(0)), dim=1)
            generator.begin_stream(input_ids, settings)
            continue
        case _:
          cogenerator.appendCurrentV(token)
    if eos:
      print()
      break
    if generated_tokens == max_new_tokens:
      printEOS("max_new_tokens")
      break

  output = cogenerator.output
  print(col_bot, output, col_default, sep="")
  yield output


def cogenerateMulti(
    cogenerator: Cogenerator,
    input_ids: torch.Tensor = empty_ids,
    max_new_tokens: int = 512,
):
  # Send prompt to generator to begin stream

  generator.begin_stream(input_ids, settings)

  # Streaming loop. Note that repeated calls to sys.stdout.flush() adds some latency, but some
  # consoles won't update partial lines without it.

  generated_tokens = 0

  printChunks(col_bot)
  while True:
    chunk, eos, chunk_tokens = generator.stream()
    if eos:
      printEOS("exllamav2")
      cogenerator.setCurrentV()
      break

    printChunks(chunk)
    sys.stdout.flush()
    for token_tensor in chunk_tokens[0]:
      generated_tokens += 1
      token = token_tensor.item()
      cogenerator.appendCurrentV(token)
    if generated_tokens == max_new_tokens:
      printEOS("max_new_tokens")
      break

  output = cogenerator.output
  print(col_bot, output, col_default, sep="")
  return output
