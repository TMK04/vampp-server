# From https://github.com/turboderp/exllamav2/discussions/94
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, Dict, List, Optional
from langchain.pydantic_v1 import Field, root_validator

from exllamav2.model import ExLlamaV2
from exllamav2.cache import ExLlamaV2Cache
from exllamav2.config import ExLlamaV2Config
from exllamav2.tokenizer import ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator
from exllamav2.generator.sampler import ExLlamaV2Sampler
import os, glob


class ExllamaV2(LLM):
  client: Any  #: :meta private:
  model_path: str
  """The path to the GPTQ model folder."""
  exllama_cache: ExLlamaV2Cache = None  #: :meta private:
  config: ExLlamaV2Config = None  #: :meta private:
  generator: ExLlamaV2StreamingGenerator = None  #: :meta private:
  tokenizer: ExLlamaV2Tokenizer = None  #: :meta private:
  settings: ExLlamaV2Sampler.Settings = None  #: :meta private:

  ##Langchain parameters
  logfunc = print
  stop_strings: Optional[List[str]] = Field(
      "", description="Sequences that immediately will stop the generator.")
  streaming: Optional[bool] = Field(True,
                                    description="Whether to stream the results, token by token.")

  ##Generator parameters
  disallowed_tokens: Optional[List[int]] = Field(
      None, description="List of tokens to disallow during generation.")
  temperature: Optional[float] = Field(None, description="Temperature for sampling diversity.")
  top_k: Optional[int] = Field(
      None, description="Consider the most probable top_k samples, 0 to disable top_k sampling.")
  top_p: Optional[float] = Field(
      None,
      description=
      "Consider tokens up to a cumulative probabiltiy of top_p, 0.0 to disable top_p sampling.")
  # min_p: Optional[float] = Field(None, description="Do not consider tokens with probability less than this.")
  typical: Optional[float] = Field(
      None, description="Locally typical sampling threshold, 0.0 to disable typical sampling.")
  token_repetition_penalty: Optional[float] = Field(
      None, description="Repetition penalty for most recent tokens.")
  token_repetition_range: Optional[int] = Field(
      None,
      description="No. most recent tokens to repeat penalty for, -1 to apply to whole context.")
  token_repetition_decay: Optional[int] = Field(
      None, description="Gradually decrease penalty over this many tokens.")
  # beams: Optional[int] = Field(None, description="Number of beams for beam search.")
  # beam_length: Optional[int] = Field(None, description="Length of beams for beam search.")

  ##Config overrides
  max_seq_len: Optional[int] = Field(
      2048,
      decription=
      "Reduce to save memory. Can also be increased, ideally while also using scale_pos_emn and a compatible model/LoRA"
  )
  scale_pos_emb: Optional[float] = Field(
      1.0,
      description=
      "Factor by which to scale positional embeddings, e.g. for 4096-token sequence use a scaling factor of 2.0, requires finetuned model or LoRA"
  )
  # alpha_value: Optional[float] = Field(1.0, description="Rope context extension alpha") #Old Param
  scale_alpha_value: Optional[float] = Field(1.0,
                                             description="Rope context extension alpha")  #New Param

  ##Lora Parameters
  # lora_path: Optional[str] = Field(None, description="Path to your lora.") #Exllamav2 doesn't yet support loras

  @staticmethod
  def get_model_path_at(path):
    patterns = ["*.safetensors", "*.bin", "*.pt"]
    model_paths = []
    for pattern in patterns:
      full_pattern = os.path.join(path, pattern)
      model_paths = glob.glob(full_pattern)
      if model_paths:  # If there are any files matching the current pattern
        break  # Exit the loop as soon as we find a matching file
    if model_paths:  # If there are any files matching any of the patterns
      return model_paths[0]
    else:
      return None  # Return None if no matching files were found

  @staticmethod
  def configure_object(params, values, logfunc):
    obj_params = {k: values.get(k) for k in params}

    def apply_to(obj):
      for key, value in obj_params.items():
        if value:
          if hasattr(obj, key):
            setattr(obj, key, value)
            logfunc(f"{key} {value}")
          else:
            raise AttributeError(f"{key} does not exist in {obj}")

    return apply_to

  @root_validator()
  def validate_environment(cls, values: Dict) -> Dict:
    model_path = values["model_path"]
    # lora_path = values["lora_path"]

    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()

    tokenizer_path = os.path.join(model_path, "tokenizer.model")
    model_config_path = os.path.join(model_path, "config.json")
    model_path = ExllamaV2.get_model_path_at(model_path)

    # config = ExLlamaV2Config(model_config_path)
    # tokenizer = ExLlamaV2Tokenizer(tokenizer_path)
    tokenizer = ExLlamaV2Tokenizer(config)
    # config.model_path = model_path

    ##Set logging function if verbose or set to empty lambda
    verbose = values['verbose']
    if not verbose:
      values['logfunc'] = lambda *args, **kwargs: None
    logfunc = values['logfunc']

    model_param_names = [
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical",
        "token_repetition_penalty_max",
        "token_repetition_penalty_sustain",
        "token_repetition_penalty_decay",
        "beams",
        "beam_length",
    ]

    config_param_names = [
        "max_seq_len",
        # "alpha_value"
        "scale_alpha_value"
    ]

    configure_config = ExllamaV2.configure_object(config_param_names, values, logfunc)
    configure_config(config)
    configure_model = ExllamaV2.configure_object(model_param_names, values, logfunc)

    model = ExLlamaV2(config)
    model.load()

    exllama_cache = ExLlamaV2Cache(model)
    settings = ExLlamaV2Sampler.Settings()
    # settings = ExLlamaV2Sampler.Settings()
    configure_model(settings)
    # settings.temperature = 0.85
    # settings.top_k = 50
    # settings.top_p = 0.8
    # settings.token_repetition_penalty = 1.15
    # settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    generator = ExLlamaV2StreamingGenerator(model, exllama_cache, tokenizer)

    ##Load and apply lora to generator
    # if lora_path is not None:
    #     lora_config_path = os.path.join(lora_path, "adapter_config.json")
    #     lora_path = ExllamaV2.get_model_path_at(lora_path)
    #     lora = ExLlamaLora(model, lora_config_path, lora_path)
    #     generator.lora = lora
    #     logfunc(f"Loaded LORA @ {lora_path}")

    ##Configure the model and generator
    values["stop_strings"] = [x.strip().lower() for x in values["stop_strings"]]

    print(generator.__dict__)

    # configure_model(generator.settings) #This may be necessary
    setattr(settings, "stop_strings", values["stop_strings"])
    logfunc(f"stop_strings {values['stop_strings']}")

    disallowed = values.get("disallowed_tokens")
    if disallowed:
      generator.disallow_tokens(disallowed)
      print(f"Disallowed Tokens: {generator.disallowed_tokens}")

    values["client"] = model
    values["generator"] = generator
    values["config"] = config
    values["tokenizer"] = tokenizer
    values["exllama_cache"] = exllama_cache
    values["settings"] = settings

    return values

  @property
  def _llm_type(self) -> str:
    """Return type of llm."""
    return "Exllama"

  def get_num_tokens(self, text: str) -> int:
    """Get the number of tokens present in the text."""
    return self.generator.tokenizer.num_tokens(text)

  def _call(
      self,
      prompt: str,
      stop: Optional[List[str]] = None,
      run_manager: Optional[CallbackManagerForLLMRun] = None,
      **kwargs: Any,
  ) -> str:
    combined_text_output = ""
    for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
      combined_text_output += token
    return combined_text_output

  from enum import Enum

  class MatchStatus(Enum):
    EXACT_MATCH = 1
    PARTIAL_MATCH = 0
    NO_MATCH = 2

  def match_status(self, sequence: str, banned_sequences: List[str]):
    sequence = sequence.strip().lower()
    for banned_seq in banned_sequences:
      if banned_seq == sequence:
        return self.MatchStatus.EXACT_MATCH
      elif banned_seq.startswith(sequence):
        return self.MatchStatus.PARTIAL_MATCH
    return self.MatchStatus.NO_MATCH

  def stream(
      self,
      prompt: str,
      stop: Optional[List[str]] = None,
      run_manager: Optional[CallbackManagerForLLMRun] = None,
  ) -> str:
    # config = self.config
    generator = self.generator
    # tokenizer = self.tokenizer
    # beam_search = (self.beams and self.beams >= 1 and self.beam_length and self.beam_length >= 1)

    ids = generator.tokenizer.encode(prompt)
    generator._gen_begin_reuse(ids, self.settings)

    # if beam_search:
    #     generator.begin_beam_search()
    #     token_getter = generator.beam_search
    # else:
    #     generator.end_beam_search()
    token_getter = generator._gen_single_token

    last_newline_pos = 0
    match_buffer = ""

    # seq_length = len(generator.tokenizer.decode(generator.sequence_actual[0])) # Old line
    seq_length = len(generator.tokenizer.decode(generator.sequence_ids[0]))
    response_start = seq_length
    cursor_head = response_start

    # while(generator.gen_num_tokens() <= (self.max_seq_len - 4)): #Slight extra padding space as we seem to occassionally get a few more than 1-2 tokens
    while (
        len(generator.sequence_ids) <= (self.max_seq_len - 4)
    ):  #Slight extra padding space as we seem to occassionally get a few more than 1-2 tokens

      #Fetch a token
      token, eos = token_getter(self.settings)

      #If it's the ending token replace it and end the generation.
      if token.item() == generator.tokenizer.eos_token_id:
        # generator.replace_last_token(generator.tokenizer.newline_token_id)
        generator.sequence_ids[:, -1] = generator.tokenizer.newline_token_id
        # if beam_search:
        #     generator.end_beam_search()
        return

      #Tokenize the string from the last new line, we can't just decode the last token due to how sentencepiece decodes.
      stuff = generator.tokenizer.decode(generator.sequence_ids[0][last_newline_pos:])
      cursor_tail = len(stuff)

      # chunk = stuff[cursor_head:cursor_tail] # Old code, fails decoding special characters.

      if (cursor_tail <
          cursor_head):  # This happens on special characters and can fail to decode them.
        chunk = stuff[-1:]
      else:
        chunk = stuff[cursor_head:cursor_tail]

      # Manually remove the unknown character token. There is an issue decoding special characters i.e. emojis.
      if (bytes(chunk, encoding="utf-8").hex() == "efbfbd"):
        chunk = ""

      cursor_head = cursor_tail

      #Append the generated chunk to our stream buffer
      match_buffer = match_buffer + chunk

      if token.item() == generator.tokenizer.newline_token_id:
        last_newline_pos = len(generator.sequence_ids[0])
        cursor_head = 0
        cursor_tail = 0

      #Check if the stream buffer is one of the stop sequences
      status = self.match_status(match_buffer, self.stop_strings)

      if status == self.MatchStatus.EXACT_MATCH:
        #Encountered a stop, rewind our generator to before we hit the match and end generation.
        rewind_length = generator.tokenizer.encode(match_buffer).shape[-1]
        generator.gen_rewind(rewind_length)
        gen = generator.tokenizer.decode(generator.sequence_ids[0][response_start:])
        # if beam_search:
        #     generator.end_beam_search()
        return
      elif status == self.MatchStatus.PARTIAL_MATCH:
        #Partially matched a stop, continue buffering but don't yield.
        continue
      elif status == self.MatchStatus.NO_MATCH:
        if run_manager:
          run_manager.on_llm_new_token(
              token=match_buffer,
              verbose=self.verbose,
          )
        yield match_buffer  # Not a stop, yield the match buffer.
        match_buffer = ""

    return
