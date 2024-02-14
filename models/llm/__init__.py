from .cogenerate import Cogenerator, ScoreCogenerator, ScoreJustificationCogenerator, SummaryCogenerator, TopicCogenerator
from .tokenizer import cleanStopStrings


def generate(content: str, topic: str, yt_title: str):
  content = cleanStopStrings(content)

  content_pretokenized = Cogenerator.PretokenizeInput(content)
  if not topic:
    topic_content_pretokenized = Cogenerator.PretokenizeInput(f"""YouTube Title: {yt_title}
  {content}""") if yt_title else content_pretokenized
    for topic in TopicCogenerator.cogenerate(topic_content_pretokenized):
      yield "topic", topic

  topic_pretokenized = Cogenerator.PretokenizeInput(topic)
  for summary in SummaryCogenerator.cogenerate(content_pretokenized, topic_pretokenized):
    yield "summary", summary
  for score_name in ScoreJustificationCogenerator.score_names:
    score_justification_key = f"{score_name}_justification"
    for score_justification in ScoreJustificationCogenerator.cogenerate(
        score_name, content_pretokenized, topic_pretokenized):
      yield score_justification_key, score_justification
    score_justification_pretokenized = Cogenerator.PretokenizeInput(score_justification)
    for score in ScoreCogenerator.cogenerate(score_name, content_pretokenized, topic_pretokenized,
                                             score_justification_pretokenized):
      yield score_name, score
