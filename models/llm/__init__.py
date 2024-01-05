from .exllamav2_loader import ExllamaV2
from .prompts import dict_h_base_h, pitch_prompt, prompt, score_parser, summary_prompt, summary_topic_prompt, summary_topic_prompt_w_title, summary_topic_parser
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory, DynamoDBChatMessageHistory
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.schema import SystemMessage
import os
from pathlib import Path

from server.aws import AWS_DYNAMO_TABLE
from server.config import MODEL_LLM_CONTEXT_LEN, MODEL_LLM_DIR, MODEL_LLM_SCALE_POS_EMB, MODEL_SD_DIR

shared_kwargs = dict(
    max_input_len=MODEL_LLM_CONTEXT_LEN,
    max_seq_len=MODEL_LLM_CONTEXT_LEN,
    scale_pos_emb=MODEL_LLM_SCALE_POS_EMB,
    stop_strings=[*dict_h_base_h.values()],
    top_p=.9,
    verbose=True,
)
models_dir = Path(__file__).parent / "./models/"
llm = ExllamaV2(
    **shared_kwargs,
    model_path=os.path.join(models_dir, MODEL_LLM_DIR),
    #streaming = True,
    # lora_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None,
    # callbacks=[
    #     handler,
    # ],
    temperature=.7,
    top_k=50,
    typical=.95,
    token_repetition_penalty=1.15,
)
sd = ExllamaV2(
    **shared_kwargs,
    model_path=os.path.join(models_dir, MODEL_SD_DIR),
    #streaming = True,
    # lora_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None,
    # callbacks=[
    #     handler,
    # ],
    temperature=.5,
    top_k=40,
    typical=.2,
    token_repetition_penalty_max=1.1,
)
score_parser = RetryWithErrorOutputParser.from_llm(parser=score_parser, llm=llm)
summary_topic_parser = RetryWithErrorOutputParser.from_llm(parser=summary_topic_parser, llm=llm)


def summarizeWithTopic(topic, pitch):
  prompt_value_str = summary_prompt.format(topic=topic, pitch=pitch)
  failures = 0
  while failures < 3:
    try:
      summary = llm(prompt_value_str)
      return summary
    except Exception as e:
      failures += 1
      str_e = str(e)
      if (str_e.endswith(f"exceeds dimension size ({MODEL_LLM_CONTEXT_LEN}).")):
        raise ValueError(str_e)
      print(str_e)
      print(prompt_value_str)
      print("Retrying...")


def summarizeWithoutTopic(pitch, title):
  if title:
    prompt_value = summary_topic_prompt_w_title.format_prompt(pitch=pitch, title=title)
  else:
    prompt_value = summary_topic_prompt.format_prompt(pitch=pitch)
  prompt_value_str = prompt_value.to_string()
  failures = 0
  while failures < 3:
    try:
      summary_topic_response_str = llm(prompt_value_str)
    except Exception as e:
      failures += 1
      str_e = str(e)
      if (str_e.endswith(f"exceeds dimension size ({MODEL_LLM_CONTEXT_LEN}).")):
        raise ValueError(str_e)
      print(str_e)
      print(prompt_value_str)
      print("Retrying...")
      continue
    try:
      summary_topic_response = summary_topic_parser.parse_with_prompt(summary_topic_response_str,
                                                                      prompt_value)
      return summary_topic_response
    except Exception as e:
      failures += 1
      str_e = str(e)
      print(str_e)
      print(summary_topic_response_str)
      print("Retrying...")


def summarize(pitch, topic, title):
  if topic:
    print(f"Summarizing with topic: {topic}")
    summary = summarizeWithTopic(topic=topic, pitch=pitch)
    return topic, summary

  print("Summarizing without topic...")
  summary_topic_response = summarizeWithoutTopic(pitch=pitch, title=title)
  return summary_topic_response.topic, summary_topic_response.summary


dict_id_chain = {}


def Chain(id):
  if id in dict_id_chain:
    return dict_id_chain[id]

  history = DynamoDBChatMessageHistory(
      table_name=AWS_DYNAMO_TABLE,
      session_id=id,
      primary_key_name="id",
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
  prompt_value = pitch_prompt.format_prompt(topic=topic, pitch=pitch)
  prompt_value_str = prompt_value.to_string()
  failures = 0
  while failures < 3:
    try:
      beholder_response_str = chain.run(input=prompt_value_str)
    except Exception as e:
      failures += 1
      str_e = str(e)
      if (str_e.endswith(f"exceeds dimension size ({MODEL_LLM_CONTEXT_LEN}).")):
        chain.memory.chat_memory.prune()
        raise ValueError(str_e)
      print(str_e)
      print(prompt_value_str)
      print("Retrying...")
      continue
    try:
      beholder_response = score_parser.parse_with_prompt(beholder_response_str, prompt_value)
      return beholder_response
    except Exception as e:
      failures += 1
      str_e = str(e)
      print(str_e)
      print(beholder_response_str)
      chain.memory.chat_memory.add_message(SystemMessage(content=str_e))
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
