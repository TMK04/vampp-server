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


append_pretokenized = """
========
Summary
* brief (max 1000 characters)
====
"""
prepend_pretokenized = """Summarize the following project pitches.""" + response_sep + f"""INPUT:
Hello, I'm Alex Turner, presenting a project to address the lack of accessible platforms for hands-on learning in space exploration. My solution is the Space Exploration Simulation Platform (SESP). This platform offers an immersive and interactive environment for enthusiasts to conduct experiments, simulate space missions, and enhance their skills. Modeled after successful concepts like Cybersecurity Capture The Flag, SESP provides a user-friendly interface, step-by-step learning, and a live scoreboard for real-time progress tracking. The goal is to empower users to bridge the gap between theoretical knowledge and practical experience, preparing them for the challenges of space exploration. Thank you for considering my proposal.{TopicCogenerator.append_short}Space Exploration Simulation Platform{append_pretokenized}Alex Turner proposes the Space Exploration Simulation Platform (SESP) to provide accessible hands-on learning in space exploration. SESP offers an immersive environment for simulating missions and experiments, bridging the gap between theory and practice.""" + response_sep + """INPUT:
"""

prepend_pretokenized = PretokenizePrepend(prepend_pretokenized)
append_pretokenized = PretokenizeAppend(append_pretokenized)


def Wrap(content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  return torch.cat(
      (prepend_pretokenized, content_pretokenized, TopicCogenerator.append_short_pretokenized,
       topic_pretokenized, append_pretokenized),
      dim=1)


def cogenerate(content_pretokenized: torch.Tensor, topic_pretokenized: torch.Tensor):
  input = Wrap(content_pretokenized, topic_pretokenized)
  cogenerator = SummaryCogenerator()
  return cogenerateMulti(cogenerator, input)
