from server.models.llm.prompts.TopicPrompt import append, prepend
from .Cogenerator import Cogenerator, PretokenizeAppend, PretokenizePrepend


class TopicCogenerator(Cogenerator):

  def __init__(self):
    super().__init__()
    self.output = None

  def setCurrentV(self):
    v = self.popCurrentV()
    self.output = v
    return True, None


prepend_pretokenized = PretokenizePrepend(prepend)
append_pretokenized = PretokenizeAppend(append)
