from typing import Dict, List, Any, Optional

from overrides import overrides
from allennlp.models.model import Model
import torch
from torch.nn.modules.linear import Linear
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.modules import TextFieldEmbedder, FeedForward
import allennlp.nn.util as util
import torch.nn.functional as F
from allennlp.data import Vocabulary

@Model.register("legal_pairwise")
class LegalPairwise(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder
                 ) -> None:
        super().__init__(vocab)
        self.vocab = vocab

        self.tag_projection_layer = Linear(vocab.get_vocab_size(), 2)

    @overrides
    def forward(self,  # type: ignore
                graf: Dict[str, torch.LongTensor],
                const: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None, # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        common = torch.zeros(self.vocab.get_vocab_size())
        # get all elements in common between graf and const
        # project to 2 dimensions, and we're done.




