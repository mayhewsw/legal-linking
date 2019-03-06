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

        self._doc_encoder = doc_encoder

        self.ff = FeedForward(self._token_embedder.get_output_dim()*2, num_layers=4,
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
        # graf_importance_scores = (similarity_matrix*const_mask.float().unsqueeze(1)).sum(dim=2)
        # print(graf_importance_scores.shape)
        # graf_importance_distribution = masked_softmax(graf_importance_scores, mask=graf_mask)
        # print(graf_importance_distribution)
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
        weighted_const = attended_const.sum(dim=1)

        # Shape: (batch_size, const_length, graf_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), graf_mask)
        # Shape: (batch_size, const_length, embedding_dim)
        attended_graf = weighted_sum(graf_emb, h2p_attention)
        # Shape: (batch_size, embedding_dim)
        weighted_graf = attended_graf.sum(dim=1)

        #print(weighted_graf.shape)
        #print(weighted_const.shape)

        aggregate_input = torch.cat([weighted_graf, weighted_const], dim=-1)
        logits = self.tag_projection_layer(self.ff(aggregate_input))

        #label_probs = torch.nn.functional.softmax(logits, dim=-1)

        #graf_doc = self._doc_encoder(graf_emb, graf_mask)
        #const_doc = self._doc_encoder(const_emb, const_mask)

        # concatenate each vector
        # shape: (batch_size, doc_emb_size*2)
        #grafconst = torch.cat([graf_doc, const_doc], dim=1)

        #logits = self.tag_projection_layer(self.ff(grafconst))

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
