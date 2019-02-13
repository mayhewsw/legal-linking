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

        self.accuracy = CategoricalAccuracy()
        self.tag_projection_layer = Linear(vocab.get_vocab_size(), 2)

    @overrides
    def forward(self,  # type: ignore
                graf: Dict[str, torch.LongTensor],
                const: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,  # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        #print(graf)
        #print(const)
        graf_toks = graf["tokens"]
        const_toks = const["tokens"]
        batch_size, max_seq_len = graf_toks.shape
        common = torch.zeros(batch_size, self.vocab.get_vocab_size())
        # get all elements in common between graf and const
        # project to 2 dimensions, and we're done.

        for b in range(batch_size):
            graf_set = set()
            common_toks = set()
            for tok in graf_toks[b]:
                if tok == 0:
                    continue
                graf_set.add(tok.item())

            for tok in const_toks[b]:
                if tok == 0:
                    continue

                if tok.item() in graf_set:
                    common_toks.add(tok.item())

            for c in common_toks:
                common[b, c] = 1

        logits = self.tag_projection_layer(common)

        logprob_logits = F.log_softmax(logits, dim=-1)

        output = {"prediction": torch.argmax(logprob_logits)}
        if label is not None:
            self.accuracy(logprob_logits, label, torch.ones(1))
            loss = -logprob_logits[:, label]
            output["loss"] = loss

        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}









