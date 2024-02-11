import torch

from server.models.llm.prompts.TopicPrompt import append, prepend
from .Parser import Parser, PretokenizeAppend, PretokenizeInput, PretokenizePrepend


class TopicParser(Parser):

  def __init__(self, input: str):
    super().__init__()
    input_pretokenized = PretokenizeInput(input)
    self.first_tokens = torch.cat((prepend_pretokenized, input_pretokenized, append_pretokenized))
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


prepend_pretokenized = PretokenizePrepend(prepend)
append_pretokenized = PretokenizeAppend(append)
