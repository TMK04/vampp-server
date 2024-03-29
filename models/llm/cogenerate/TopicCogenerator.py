import torch

from server.models.llm.cogenerate.fns import cogenerateSingle
from server.models.llm.tokenizer import response_sep

from .Cogenerator import Cogenerator, PretokenizeAppend, PretokenizePrepend


class TopicCogenerator(Cogenerator):

  def __init__(self):
    super().__init__()
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


append_short = """

RESPONSE:
Topic
====
"""
append_short_pretokenized = PretokenizeAppend(append_short)
prepend_pretokenized = PretokenizePrepend("""Summarize the following project pitches.""" +
                                          response_sep + f"""INPUT:
Example Pitch{append_short}Example Topic""" + response_sep + """INPUT:
""")

append_pretokenized = PretokenizeAppend("""

RESPONSE:
Topic (3-10 words)
====
""")


def Wrap(content_pretokenized: torch.Tensor):
  return torch.cat((prepend_pretokenized, content_pretokenized, append_pretokenized), dim=1)


def cogenerate(content_pretokenized: torch.Tensor):
  input = Wrap(content_pretokenized)
  cogenerator = TopicCogenerator()
  return cogenerateSingle(cogenerator, input, 32)
