import torch

from server.models.llm.prompts.TopicPrompt import append, prepend
from .Parser import Parser, PretokenizeStart, PretokenizeContinue


class TopicParser(Parser):

  def __init__(self, input: str):
    super().__init__()
    input_pretokenized = PretokenizeStart(input)
    print(prepend_pretokenized.shape, input_pretokenized.shape, append_pretokenized.shape)
    self.first_tokens = torch.cat((prepend_pretokenized, input_pretokenized, append_pretokenized))
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


append_pretokenized = PretokenizeContinue(append)
prepend_pretokenized = PretokenizeStart(prepend)
