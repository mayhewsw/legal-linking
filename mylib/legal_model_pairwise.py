from typing import Dict, List, Any, Optional

from allennlp.modules.matrix_attention import DotProductMatrixAttention
from overrides import overrides
from allennlp.models.model import Model
from allennlp.modules.token_embedders import BagOfWordCountsTokenEmbedder
import torch
from torch.nn.modules.linear import Linear
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder
import allennlp.nn.util as util
from allennlp.nn import Activation
import torch.nn.functional as F
from allennlp.data import Vocabulary

from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values


@Model.register("legal_pairwise")
class LegalPairwise(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 doc_encoder: Seq2VecEncoder,
                 ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size("tokens")

        self.metric = F1Measure(positive_label=1)

        self._token_embedder = text_field_embedder

        self._matrix_attention = DotProductMatrixAttention()

        self._compare_feedforward = FeedForward(self._token_embedder.get_output_dim()*2,
                                                num_layers=4,
                                                hidden_dims=100,
                                                activations=Activation.by_name("relu")())

        self._doc_encoder = doc_encoder

        self.ff = FeedForward(self._compare_feedforward.get_output_dim()*2, num_layers=4,
                              hidden_dims=100,
                              activations=Activation.by_name("relu")())

        self.tag_projection_layer = Linear(self.ff.get_output_dim(), 2)

    @overrides
    def forward(self,  # type: ignore
                graf: Dict[str, torch.LongTensor],
                const: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,  # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, seq_len, vocab_size)
        graf_emb = self._token_embedder(graf)
        const_emb = self._token_embedder(const)

        #print()
        #print(metadata)
        #print("ge", graf_emb.shape)
        #print("ce", const_emb.shape)

        graf_mask = util.get_text_field_mask(graf)
        const_mask = util.get_text_field_mask(const)

        # shape: (batch_size, graf_length, const_length)
        similarity_matrix = self._matrix_attention(graf_emb, const_emb)

        #print("sim", similarity_matrix.shape)

        # shape: (batch_size, graf_len)
        graf_importance_scores = (similarity_matrix*const_mask.float().unsqueeze(1)).sum(dim=2)
        graf_importance_distribution = masked_softmax(graf_importance_scores, mask=graf_mask)

        #
        # # shape (batch_size, const_len)
        # const_importance_scores = (similarity_matrix*graf_mask.float().unsqueeze(1)).sum(dim=2)
        # print(const_importance_scores.shape)
        # const_importance_distribution = masked_softmax(const_importance_scores, mask=const_mask)
        # print(const_importance_distribution)

        # graf: premise
        # cosnt: hypothesis

        # Shape: (batch_size, graf_length, const_length)
        p2h_attention = masked_softmax(similarity_matrix, const_mask)
        #print(p2h_attention.shape)
        # Shape: (batch_size, graf_length, embedding_dim)
        attended_const = weighted_sum(const_emb, p2h_attention)
        # Shape: (batch_size, embedding_dim)
        # weighted_const = attended_const.sum(dim=1)

        # Shape: (batch_size, const_length, graf_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), graf_mask)
        # Shape: (batch_size, const_length, embedding_dim)
        attended_graf = weighted_sum(graf_emb, h2p_attention)
        # at this point, every row in attended_graf is a weighted combination of words (vectors) in const.
        # Shape: (batch_size, embedding_dim)
        # weighted_graf = attended_graf.sum(dim=1)
        # get a single word vector that is a mix of all vectors in const.

        graf_compare_input = torch.cat([graf_emb, attended_const], dim=-1)
        const_compare_input = torch.cat([const_emb, attended_graf], dim=-1)

        # Shape: (batch_size, premise_length, embedding_dim)
        compared_graf = self._compare_feedforward(graf_compare_input)
        compared_graf = compared_graf * graf_mask.float().unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        weighted_premise = True
        if weighted_premise:
            compared_graf = (compared_graf*graf_importance_distribution.unsqueeze(2)).sum(dim=1)
        else:
            compared_graf = compared_graf.sum(dim=1)

        # Shape: (batch_size, hypothesis_length, embedding_dim)
        compared_const = self._compare_feedforward(const_compare_input)
        compared_const = compared_const * const_mask.float().unsqueeze(-1)

        # Shape: (batch_size, compare_dim)
        compared_const = compared_const.sum(dim=1)

        aggregate_input = torch.cat([compared_graf, compared_const], dim=-1)
        logits = self.tag_projection_layer(self.ff(aggregate_input))

        logprob_logits = F.log_softmax(logits, dim=-1)
        class_probabilities = torch.exp(logprob_logits)
        label_predictions = torch.argmax(logprob_logits, dim=-1)
        prediction_probs = torch.gather(class_probabilities, 1, label_predictions.unsqueeze(-1))

        output = {"prediction": label_predictions, "prediction_prob": prediction_probs}
        if label is not None:
            self.metric(logprob_logits, label, torch.ones(1))
            logprob = torch.gather(logprob_logits, 1, label.unsqueeze(-1))

            loss = -logprob.sum()
            output["loss"] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        prec, rec, f1 = self.metric.get_metric(reset=reset)
        return {"precision": prec, "recall": rec, "f1": f1}
