from typing import Dict, List, Any, Optional

from overrides import overrides
from allennlp.models.model import Model
from allennlp.modules.token_embedders import BagOfWordCountsTokenEmbedder
import torch
from torch.nn.modules.linear import Linear
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure
from allennlp.modules import TextFieldEmbedder, FeedForward
import allennlp.nn.util as util
from allennlp.nn import Activation
import torch.nn.functional as F
from allennlp.data import Vocabulary


@Model.register("legal_pairwise")
class LegalPairwise(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder
                 ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size("tokens")

        #self.accuracy = CategoricalAccuracy()
        self.metric = F1Measure(positive_label=1)

        self._embedder = BagOfWordCountsTokenEmbedder(self.vocab, "tokens")

        # the last layer of this is intended to be a tag projection layer.
        # self.ff = FeedForward(self.vocab_size*2, num_layers=4,
        #                       hidden_dims=[self.vocab_size, int(self.vocab_size/2.), int(self.vocab_size/4.), 2],
        #                       activations=Activation.by_name("relu"))
        self.ff = FeedForward(self.vocab_size*2, num_layers=4,
                              hidden_dims=500,
                              activations=Activation.by_name("relu")())

        self.tag_projection_layer = Linear(self.ff.get_output_dim(), 2)

    @overrides
    def forward(self,  # type: ignore
                graf: Dict[str, torch.LongTensor],
                const: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,  # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, vocab_size)
        graf_emb = self._embedder(graf["tokens"])
        const_emb = self._embedder(const["tokens"])

        # concatenate each vector
        # shape: (batch_size, vocab_size*2)
        grafconst = torch.cat([graf_emb, const_emb], dim=1)

        #print(grafconst)

        #graf_toks = graf["tokens"]
        #const_toks = const["tokens"]
        #batch_size, max_seq_len = graf_toks.shape
        #common = torch.zeros(batch_size, self.vocab.get_vocab_size())
        # get all elements in common between graf and const
        # project to 2 dimensions, and we're done.

        logits = self.tag_projection_layer(self.ff(grafconst))

        logprob_logits = F.log_softmax(logits, dim=-1)
        class_probabilities = torch.exp(logprob_logits)
        label_predictions = torch.argmax(logprob_logits, dim=-1)
        prediction_probs = torch.gather(class_probabilities, 1, label_predictions.unsqueeze(-1))

        output = {"prediction": label_predictions, "prediction_prob" : prediction_probs}
        if label is not None:
            self.metric(logprob_logits, label, torch.ones(1))
            logprob = torch.gather(logprob_logits, 1, label.unsqueeze(-1))

            loss = -logprob.sum()
            output["loss"] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        prec, rec, f1 = self.metric.get_metric(reset=reset)
        return {"precision": prec, "recall": rec, "f1": f1}
