import sys
import torch
from exllamav2.generator import ExLlamaV2Sampler

from server.models.llm.generator import generator
from server.models.llm.tokenizer import tokenizer
from server.models.llm.utils import col_bot, col_default, printChunks, printEOS, printPrompt

from .Cogenerator import Cogenerator

from typing import Callable

settings = ExLlamaV2Sampler.Settings()
settings.token_repetition_penalty = 1.1
settings.top_k = 1


def cogenerate(_iter: Callable):

  def cogenerate(cogenerator: Cogenerator, input_ids: torch.Tensor, max_new_tokens: int):
    nonlocal _iter

    printPrompt(tokenizer.decode(input_ids, decode_special_tokens=True)[0])

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

      _break, _generated_tokens = _iter(cogenerator, chunk_tokens)

      if _break:
        break

      generated_tokens += _generated_tokens
      if generated_tokens >= max_new_tokens:
        printEOS("max_new_tokens")
        break

      chunk_prev += chunk
      yield chunk_prev

    output = cogenerator.output
    if output is not None:
      print(col_bot, output, col_default, sep="")
      yield output

  return cogenerate


def iterSingle(cogenerator: Cogenerator, chunk_tokens: torch.Tensor):
  global generator

  generated_tokens = 0
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
            printEOS("done\n")
            return True, 0
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
  return False, generated_tokens


def iterMulti(cogenerator: Cogenerator, chunk_tokens: torch.Tensor):
  generated_tokens = 0
  for token_tensor in chunk_tokens[0]:
    generated_tokens += 1
    token = token_tensor.item()
    cogenerator.appendCurrentV(token)
  return False, generated_tokens


cogenerateSingle = cogenerate(iterSingle)
cogenerateMulti = cogenerate(iterMulti)
