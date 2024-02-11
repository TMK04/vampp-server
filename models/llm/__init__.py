from .generates.generateYAML import generateYAML
from .parsers.TopicParser import TopicParser


def generateTopic(content: str):
  parser = TopicParser(content)
  topic = generateYAML(parser)
  return topic
