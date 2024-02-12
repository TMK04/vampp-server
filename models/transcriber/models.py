from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.nlp.models import PunctuationCapitalizationModel

transcriber = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
punctuation_restorer = PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")
