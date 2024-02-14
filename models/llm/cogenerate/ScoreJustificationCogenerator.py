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


def Prepend(score_name: str, score_justification_points: str, score_justification_eg: str):
  input_prefix = f"""Guiding questions:
{score_justification_points}

INPUT:
"""
  return f"""Analyze the following project pitches based on {score_name}.
Use all the provided Guiding questions (+ for good, - for bad).
Do NOT consider or use new questions.
Do NOT summarize or generate follow-up questions.""" + response_sep + f"""{input_prefix}Example Pitch{TopicCogenerator.append_short}Example Topic{dict_append_short[score_name]}{score_justification_eg}""" + response_sep + input_prefix


dict_prepend = {
    "Clarity":
    dict(
        points="""1. Is the language used **clear and concise**? (+)
2. Are key points illustrated with *good* examples? (+)
3. Are there *coherent* transitions between sections? (+)
4. Does the presenter use *jargon* without explaining them? (-)""",
        eg=
        "The pitch is concise with examples, but does not flow smoothly. Moreover, the presenter uses a lot of technical terms but only explains a few."
    ),
    "Creativity":
    dict(points="""1. Do similar projects exist? (-)
2. Does the project offer **unique** features that set it apart? (+)
3. Does the project utilize new ideas/technologies/etc.? (+)""",
         eg="The project is not new, but uses a new technology."),
    "Feasibility":
    dict(points="""1. Are resources **available** to implement the project? (+)
2. Are there risks in implementing & maintaining the project? (-)
3. Does the presenter have a plan to *mitigate* the risks? (+)""",
         eg="The project requires resources that are abundant, but risks are not mentioned."),
    "Impact":
    dict(points="""1. Does the project target a *specific* audience? (+)
2. Does the project **solve** the target audience's needs? (+)
3. Is the project **difficult to use** for the target audience? (-)""",
         eg="The project positively impacts its target audience, but is difficult to use."),
}
dict_prepend_pretokenized = {
    score_name:
    PretokenizePrepend(
        Prepend(score_name, score_justification_v["points"], score_justification_v["eg"]))
    for score_name, score_justification_v in dict_prepend.items()
}


def Append(score_name: str):
  return f"""
========
{score_name} Justification (20-60 words)
====
"""


dict_append_pretokenized = {
    score_name: PretokenizeAppend(Append(score_name))
    for score_name in score_names
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
