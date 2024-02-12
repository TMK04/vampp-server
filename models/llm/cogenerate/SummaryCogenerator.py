import torch

from server.models.llm.cogenerate.fns import cogenerateMulti
from server.models.llm.tokenizer import response_sep

from .Cogenerator import Cogenerator, PretokenizeAppend, PretokenizePrepend
from . import TopicCogenerator


class SummaryCogenerator(Cogenerator):

  def __init__(self):
    super().__init__()
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


append_short = """
========
Summary
====
"""

prepend_pretokenized = PretokenizePrepend("""Summarize the following project pitches.""" +
                                          response_sep + f"""INPUT:
Example Pitch{TopicCogenerator.append_short}Example Topic{append_short}Example Summary""" +
                                          response_sep + """INPUT:
""")

append_pretokenized = PretokenizeAppend("""
========
Summary
* brief (20-60 words)
====
""")


def Wrap(content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  return torch.cat(
      (prepend_pretokenized, content_pretokenized, TopicCogenerator.append_short_pretokenized,
       topic_pretokenized, append_pretokenized),
      dim=1)


def cogenerate(content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  input = Wrap(content_pretokenized, topic_pretokenized)
  cogenerator = SummaryCogenerator()
  return cogenerateMulti(cogenerator, input, 512)
