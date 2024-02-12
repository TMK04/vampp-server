import torch

from server.models.llm.cogenerate.fns import cogenerateMulti
from server.models.llm.tokenizer import response_sep

from .Cogenerator import Cogenerator, PretokenizeAppend, PretokenizePrepend
from . import TopicCogenerator

score_names = {"Clarity", "Creativity", "Feasibility", "Impact"}


class ScoreJustificationCogenerator(Cogenerator):

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
{score_name} Justification
====
"""


dict_append_short = {score_name: AppendShort(score_name) for score_name in score_names}


def Prepend(score_name: str, score_justification_eg: str):
  return f"""Analyze the following project pitches based on {score_name}.
Use all the provided guiding questions: (+ for good, - for bad)
Do NOT come up with new guiding questions.""" + response_sep + f"""INPUT:
Example Pitch{TopicCogenerator.append_short}Example Topic{dict_append_short[score_name]}{score_justification_eg}""" + response_sep + """INPUT:
"""


dict_prepend = {
    "Clarity": "The pitch uses clear language.",
    "Creativity": "The pitch shows creativity",
    "Feasibility": "The project is feasible",
    "Impact": "The project positively impacts the target audience."
}
dict_prepend_pretokenized = {
    score_name: PretokenizePrepend(Prepend(score_name, score_justification_eg))
    for score_name, score_justification_eg in dict_prepend.items()
}


def Append(score_name: str, points: str):
  return f"""
========
{score_name} Justification
* concise (20-60 words)
{points}
====
"""


dict_append_pretokenized = {
    "Clarity":
    """+ Is the language used **clear and concise**?
+ Are key points illustrated with *good* examples?
+ Are there *coherent* transitions between sections?
- Does the presenter use *jargon* without explaining them?""",
    "Creativity":
    """- Do similar projects exist?
+ Does the project offer **unique** features that set it apart?
+ Does the project utilize new ideas/technologies/etc.?""",
    "Feasibility":
    """+ Are resources **available** to implement the project?
- Are there risks in implementing & maintaining the project?
+ Does the presenter have a plan to *mitigate* the risks?""",
    "Impact":
    """+ Does the project target a *specific* audience?
+ Does the project **solve** the target audience's needs?
- Is the project **difficult to use** for the target audience?""",
}
dict_append_pretokenized = {
    score_name: PretokenizeAppend(Append(score_name, score_justification_points))
    for score_name, score_justification_points in dict_append_pretokenized.items()
}


def Wrap(score_name: str, content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  return torch.cat((dict_prepend_pretokenized[score_name], content_pretokenized,
                    TopicCogenerator.append_short_pretokenized, topic_pretokenized,
                    dict_append_pretokenized[score_name]),
                   dim=1)


def cogenerate(score_name: str, content_pretokenized: torch.Tensor,
               topic_pretokenized: torch.Tensor):
  input = Wrap(score_name, content_pretokenized, topic_pretokenized)
  cogenerator = ScoreJustificationCogenerator()
  return cogenerateMulti(cogenerator, input, 512)
