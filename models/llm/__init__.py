from .generates.generateYAML import generateYAML
from .parsers.Parser import PretokenizeAndWrap
from .parsers.TopicParser import TopicParser, append_pretokenized, prepend_pretokenized


def generateTopic(content: str):
  content_pretokenized = PretokenizeAndWrap(prepend_pretokenized, content, append_pretokenized)
  parser = TopicParser()
  topic = generateYAML(parser, content_pretokenized)
  return topic
