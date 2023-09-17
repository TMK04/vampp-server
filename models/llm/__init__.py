from .exllama_loader import Exllama
from .prompts import dict_h_base_h, pitch_prompt, prompt, score_parser
from config import MODEL_LLM_CONTEXT_LEN, MODEL_LLM_DYNAMO_HISTORY_TABLE, MODEL_LLM_PATH, MODEL_LLM_GS
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory, DynamoDBChatMessageHistory
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.schema import SystemMessage
import numpy as np
import pandas as pd
import re
import torch

shared_kwargs = dict(
    stop_sequences=[*dict_h_base_h.values()],
    max_seq_len=MODEL_LLM_CONTEXT_LEN,
    max_input_len=MODEL_LLM_CONTEXT_LEN,
    compress_pos_emb=1.0,
    top_p=.9,
    matmul_recons_thd=8,
    fused_mlp_thd=2,
    sdp_thd=8,
    fused_attn=True,
)
llm = Exllama(
    **shared_kwargs,
    #streaming = True,
    model_path=MODEL_LLM_PATH,
    # lora_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None,
    # callbacks=[
    #     handler,
    # ],
    verbose=True,
    gpu_split=MODEL_LLM_GS,
    temperature=.7,
    top_k=50,
    typical=.95,
    token_repetition_penalty_max=1.15,
)
score_parser = RetryWithErrorOutputParser.from_llm(parser=score_parser, llm=llm)

dict_id_chain = {}


def Chain(id):
  if id in dict_id_chain:
    return dict_id_chain[id]

  history = DynamoDBChatMessageHistory(
      table_name=MODEL_LLM_DYNAMO_HISTORY_TABLE,
      session_id=id,
  )
  memory = ConversationSummaryBufferMemory(llm=llm,
                                           chat_memory=history,
                                           max_token_limit=512,
                                           ai_prefix="Beholder",
                                           human_prefix="User")
  chain = ConversationChain(
      llm=llm,
      prompt=prompt,
      memory=memory,
  )
  dict_id_chain[id] = chain
  return chain


def runBeholderFirst(chain, topic, pitch):
  chain.memory.clear()
  prompt_value = pitch_prompt.format_prompt(topic=topic, pitch=pitch),
  failures = 0
  while failures < 3:
    try:
      beholder_response_str = chain.run(input=prompt_value)
    except Exception as e:
      failures += 1
      str_e = str(e)
      if (str_e.endswith(f"exceeds dimension size ({MODEL_LLM_CONTEXT_LEN}).")):
        chain.memory.chat_memory.prune()
        raise ValueError(f"Input length exceeds the maximum length of {MODEL_LLM_CONTEXT_LEN}.")
      print(str_e)
      print(beholder_kwargs)
      print("Retrying...")
      continue
    try:
      beholder_response = score_parser.parse_with_prompt(beholder_response_str, prompt_value)
      return beholder_response
    except Exception as e:
      failures += 1
      print(e)
      print(beholder_response_str)
      print("Retrying...")
  raise ValueError("Failed to parse response from Beholder.")


def runBeholder(chain, user_input):
  while failures < 3:
    try:
      beholder_response_str = chain.run(input=user_input)
    except Exception as e:
      failures += 1
      str_e = str(e)
      if (str_e.endswith(f"exceeds dimension size ({MODEL_LLM_CONTEXT_LEN}).")):
        chain.memory.chat_memory.prune()
        raise ValueError(f"Input length exceeds the maximum length of {MODEL_LLM_CONTEXT_LEN}.")
      print(str_e)
      print(beholder_kwargs)
      print("Retrying...")
      continue
  return beholder_response_str