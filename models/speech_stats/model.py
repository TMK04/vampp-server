import os``
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

from server.config import MODELS_DIR
from ..components import _Classifier, device


class RegressionHead(nn.Module):
  r"""Classification head."""

  def __init__(self, config):

    super().__init__()

    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.dropout = nn.Dropout(config.final_dropout)
    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

  def forward(self, features, **kwargs):

    x = features
    x = self.dropout(x)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)

    return x


def calcHiddenStates(x):
  return torch.mean(x[0], dim=1)


class EmotionModel(Wav2Vec2PreTrainedModel):
  r"""Speech emotion classifier."""

  def __init__(self, config):

    super().__init__(config)

    self.config = config
    self.wav2vec2 = Wav2Vec2Model(config)
    self.classifier = RegressionHead(config)
    self.init_weights()

  def forward(
      self,
      input_values,
  ):

    outputs = self.wav2vec2(input_values)
    hidden_states = calcHiddenStates(outputs)
    logits = self.classifier(hidden_states)

    return hidden_states, logits


class Head(nn.Module):

  def __init__(self, num_features, hidden_size, dropout):
    super().__init__()
    self.classifier = _Classifier(num_features, 2, hidden_size)

  def forward(self, x):
    hidden_states = calcHiddenStates(x)
    logits = self.classifier(hidden_states)
    return logits


wd = os.path.join(MODELS_DIR, "speech_stats")
model_ss_pretrained_path = os.path.join(wd, "pretrained")
model_ss_path = os.path.join(wd, "models", "model.pth")

processor = Wav2Vec2Processor.from_pretrained(model_ss_pretrained_path)

emotion_model = EmotionModel.from_pretrained(model_ss_pretrained_path)
speech_stats_model = nn.Sequential(
    emotion_model.wav2vec2,
    Head(emotion_model.config.hidden_size, 0, emotion_model.config.final_dropout)).to(device)
speech_stats_model.load_state_dict(torch.load(model_ss_path)["model"])
