import torch

from .cogenerate import TopicCogenerator, cogenerateSingle
from .cogenerate.Cogenerator import PretokenizeInput


def generate(content: str):
  content_pretokenized = PretokenizeInput(content)
  for topic in generateTopic(content_pretokenized):
    yield "topic", topic
  # for summary in generateSummary(content_pretokenized, topic):
  #   yield "summary", summary


def generateTopic(content_pretokenized: torch.Tensor):
  input = TopicCogenerator.Wrap(content_pretokenized)
  cogenerator = TopicCogenerator.TopicCogenerator()
  return cogenerateSingle(cogenerator, input)


# def generateSummary(content_pretokenized: torch.Tensor):
#   input = torch.cat((SummaryCogenerator.prepend_pretokenized, content_pretokenized,
#                      SummaryCogenerator.append_pretokenized),
#                     dim=1)
#   cogenerator = SummaryCogenerator.SummaryCogenerator()
#   return cogenerateMulti(cogenerator, content_pretokenized)

# Test
for kv in generate(
    "Good morning, I'm Hui Yang and this is my pitching video for my major project. So my business problem or business challenge is that Temasek Poly students world skills for cyber security have performed below expectations during the competition itself, and one reason being that they lack the resources to practice and hone their skills at home or in school, and also they are unable to check on the mistakes they made during the competition, as most of the time one cannot look back on which questions they got wrong, and as such, the solution that I've came up with is to create a Ctf or capture the flag, with the sole aim to train and better equip the participants as they prepare for the upcoming competition. The reason why I chose Ctft to run my Ctf is that it will be easy to navigate and it has a comprehensive guide to allow participants to see their statistics for each question that they have attempted. Ctft. Side is also a scoreboard that is run alongside with the Ctf that I have created to allow the Wss students to better track their progress. The business impacts of my project will allow the Wss students to improve on their skills and be more prepared for the competition as well as find their weak points, and with that I have end my pitching video. Thank you."
):
  print(kv)
