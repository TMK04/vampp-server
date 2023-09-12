from .exllama_loader import Exllama
from .prompts import assistant_parser, dict_h_base_h, pitch_prompt, prompt
from config import MODEL_LLM_PATH, TEXT_CONTEXT_LEN
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage
import numpy as np
import pandas as pd
import re
import torch

llm = Exllama(
    #streaming = True,
    model_path=MODEL_LLM_PATH,
    # lora_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None,
    stop_sequences=[
        *dict_h_base_h.values(),
    ],
    # callbacks=[
    #     handler,
    # ],
    verbose=True,
    set_auto_map="64",
    max_seq_len=TEXT_CONTEXT_LEN,
    max_input_len=TEXT_CONTEXT_LEN,
    compress_pos_emb=1.0,
    temperature=.7,
    top_k=50,
    top_p=.9,
    typical=.95,
    token_repetition_penalty_max=1.15,
    matmul_recons_thd=8,
    fused_mlp_thd=2,
    sdp_thd=8,
    fused_attn=True,
)

memory = ConversationSummaryBufferMemory(llm=llm,
                                         max_token_limit=256,
                                         ai_prefix="Assistant",
                                         human_prefix="User")
beholder_chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)


def runBeholderFirst(topic, pitch):
  beholder_kwargs = {
      "input": pitch_prompt.format(topic=topic, pitch=pitch),
  }
  failures = 0
  while failures < 3:
    try:
      beholder_response_str = beholder_chain.run(input=beholder_kwargs["input"])
    except Exception as e:
      failures += 1
      str_e = str(e)
      if (str_e.endswith(f"exceeds dimension size ({TEXT_CONTEXT_LEN}).")):
        beholder_chain.memory.chat_memory.prune()
        raise ValueError(f"Input length exceeds the maximum length of {TEXT_CONTEXT_LEN}.")
      print(str_e)
      print(beholder_kwargs)
      print("Retrying...")
      continue
    try:
      beholder_response = assistant_parser.parse(beholder_response_str)
      return beholder_response
    except Exception as e:
      failures += 1
      print(e)
      print(beholder_response_str)
      beholder_chain.memory.chat_memory.add_message(SystemMessage(content=str(e)))
      print("Retrying...")
  raise ValueError("Failed to parse response from Beholder.")


def runBeholder(user_input):
  while failures < 3:
    try:
      beholder_response_str = beholder_chain.run(input=user_input)
    except Exception as e:
      failures += 1
      str_e = str(e)
      if (str_e.endswith(f"exceeds dimension size ({TEXT_CONTEXT_LEN}).")):
        beholder_chain.memory.chat_memory.prune()
        raise ValueError(f"Input length exceeds the maximum length of {TEXT_CONTEXT_LEN}.")
      print(str_e)
      print(beholder_kwargs)
      print("Retrying...")
      continue
  return beholder_response_str