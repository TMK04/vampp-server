import torch

from server.models.llm.cogenerate.fns import cogenerateMulti
from server.models.llm.tokenizer import response_sep

from .Cogenerator import Cogenerator, PretokenizeAppend, PretokenizePrepend
from . import TopicCogenerator


class ScoreJustificationCogenerator(Cogenerator):

  def __init__(self):
    super().__init__()
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


def ScoreJustificationAppend(score_name: str, points: str):
  return f"""
[SEP]
{score_name} Justification
{points}
====
"""


score_justification_appends = {
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
score_justification_appends = {
    score_name: ScoreJustificationAppend(score_name, score_justification_points)
    for score_name, score_justification_points in score_justification_appends.items()
}


def ScoreJustificationPrepend(score_name: str, eg: str):
  return f"""Analyze the following pitches based on {score_name}.
Use all guiding questions (+ for good, - for bad)""" + response_sep + f"""INPUT:
Hello, I'm Alex Turner, presenting a project to address the lack of accessible platforms for hands-on learning in space exploration. My solution is the Space Exploration Simulation Platform (SESP). This platform offers an immersive and interactive environment for enthusiasts to conduct experiments, simulate space missions, and enhance their skills. Modeled after successful concepts like Cybersecurity Capture The Flag, SESP provides a user-friendly interface, step-by-step learning, and a live scoreboard for real-time progress tracking. The goal is to empower users to bridge the gap between theoretical knowledge and practical experience, preparing them for the challenges of space exploration. Thank you for considering my proposal.{TopicCogenerator.append_short_raw}Space Exploration Simulation Platform{score_justification_appends[score_name]}{eg}""" + response_sep + """INPUT:
"""


prepends = {
    "Clarity":
    """* The language is clear and concise, effectively conveying main points.
* Key points are supported with relevant examples, e.g., comparing SESP to Cybersecurity Capture The Flag.
* Transitions between sections are coherent, guiding the audience through the pitch smoothly.
* No jargon is used without explanation, ensuring audience understanding.""",
    "Creativity":
    """* Introduces a unique concept, the Space Exploration Simulation Platform (SESP).
* SESP is distinctive with its immersive environment, inspired by successful concepts like Cybersecurity Capture The Flag.
* The live scoreboard for real-time progress tracking is a novel addition.
* Creatively addresses the lack of accessible platforms for hands-on learning by bridging the gap between theory and practical experience in space exploration.""",
    "Feasibility":
    """* The success of the Space Exploration Simulation Platform (SESP) depends on the availability of resources and addressing potential technical challenges.
* Risks include technical complexities in creating an immersive environment and maintaining a live scoreboard.
* The presenter needs a clear plan to secure resources, mitigate risks, and ensure user adoption.
* Details on development, testing, and ongoing maintenance are crucial, as well as potential partnerships to enhance feasibility.""",
    "Impact":
    """* Targets space exploration enthusiasts.
* Addresses the need for accessible hands-on learning in space exploration.
* Appears to be user-friendly for the target audience.""",
}
prepends = {
    score_name: ScoreJustificationPrepend(score_name, score_justification_eg)
    for score_name, score_justification_eg in prepends.items()
}

prepends = {
    score_name: PretokenizePrepend(score_justification_prepend)
    for score_name, score_justification_prepend in prepends.items()
}
score_justification_appends = {
    score_name: PretokenizeAppend(score_justification_append)
    for score_name, score_justification_append in score_justification_appends.items()
}


def Wrap(score_name: str, content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  return torch.cat((prepends[score_name], content_pretokenized, TopicCogenerator.append_short,
                    topic_pretokenized, score_justification_appends[score_name]),
                   dim=1)


def cogenerate(score_name: str, content_pretokenized: torch.Tensor,
               topic_pretokenized: torch.Tensor):
  input = Wrap(score_name, content_pretokenized, topic_pretokenized)
  cogenerator = ScoreJustificationCogenerator()
  return cogenerateMulti(cogenerator, input)
