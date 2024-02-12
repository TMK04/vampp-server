import torch

from server.models.llm.cogenerate.fns import cogenerateSingle
from server.models.llm.tokenizer import response_sep

from .Cogenerator import Cogenerator, PretokenizeAppend, PretokenizePrepend
from . import ScoreJustificationCogenerator, TopicCogenerator


class ScoreCogenerator(Cogenerator):

  def __init__(self):
    super().__init__()
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


def AppendShort(score_name: str):
  return f"""
========
{score_name}
====
"""


dict_append_short = {
    score_name: AppendShort(score_name)
    for score_name in ScoreJustificationCogenerator.score_names
}


def Prepend(score_name: str):
  return f"""Analyze the following project pitches based on {score_name}.""" + response_sep + f"""INPUT:
Example Pitch{TopicCogenerator.append_short}Example Topic{ScoreJustificationCogenerator.dict_append_short[score_name]}Example Justification{dict_append_short[score_name]}1""" + response_sep + """INPUT:
"""


dict_prepend_pretokenized = {
    score_name: PretokenizePrepend(Prepend(score_name))
    for score_name in ScoreJustificationCogenerator.score_names
}


def Append(score_name: str):
  return f"""
========
{score_name}
* score between 1-10
====
"""


dict_append_pretokenized = {
    score_name: PretokenizeAppend(Append(score_name))
    for score_name in ScoreJustificationCogenerator.score_names
}


def Wrap(score_name: str, content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor,
         score_justification_pretokenized: torch.Tensor):
  return torch.cat((dict_prepend_pretokenized[score_name], content_pretokenized,
                    TopicCogenerator.append_short_pretokenized, topic_pretokenized,
                    ScoreJustificationCogenerator.dict_append_short_pretokenized[score_name],
                    score_justification_pretokenized, dict_append_pretokenized[score_name]),
                   dim=1)


def cogenerate(score_name: str, content_pretokenized: torch.Tensor,
               topic_pretokenized: torch.Tensor, score_justification_pretokenized: torch.Tensor):
  input = Wrap(score_name, content_pretokenized, topic_pretokenized,
               score_justification_pretokenized)
  cogenerator = ScoreCogenerator()
  return cogenerateSingle(cogenerator, input, 4)
