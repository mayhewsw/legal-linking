from typing import Dict, List, Any, Optional

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from overrides import overrides
from allennlp.models.model import Model
from allennlp.modules.token_embedders import BagOfWordCountsTokenEmbedder
import torch
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from torch.nn.modules.linear import Linear
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder
import allennlp.nn.util as util
from allennlp.nn import Activation
import torch.nn.functional as F
from allennlp.data import Vocabulary
from mylib.json2lines import JsonConverter

from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values


@Model.register("legal_classifier")
class LegalClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 doc_encoder: Seq2VecEncoder,
                 const_path: str
                 ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size("tokens")
        self.num_tags = vocab.get_vocab_size("labels")

        # I actually want to use the one from the config, but not sure how to do that.
        _spacy_word_splitter = SpacyWordSplitter()
        token_indexer = SingleIdTokenIndexer(namespace="tokens", lowercase_tokens=True,
                                             end_tokens=["@@pad@@", "@@pad@@", "@@pad@@", "@@pad@@"])
        # TODO: turn this into a parameter
        self.bow_embedder = BagOfWordCountsTokenEmbedder(vocab, "tokens")

        jc = JsonConverter()
        const, links = jc._read_const(const_path)

        assert self.num_tags == len(const) + 1

        # the extra 1 is for the NONE label.
        self.const_mat = torch.LongTensor(self.num_tags, self.bow_embedder.get_output_dim())
        if torch.cuda.is_available():
            self.const_mat = self.const_mat.cuda()

        # create the constitution matrix. Every element is one of the groups.
        tagmap = self.vocab.get_index_to_token_vocabulary("labels")
        for i in range(self.num_tags):
            tagname = tagmap[i]
            if tagname != "unmatched":
                const_text = const[tagname]
            else:
                const_text = "@@pad@@"

            const_toks = _spacy_word_splitter.split_words(const_text)
            const_indices = token_indexer.tokens_to_indices(const_toks, vocab, "tokens")
            tens = torch.LongTensor(const_indices["tokens"]).unsqueeze(0)
            self.const_mat[i, :] = self.bow_embedder(tens).clamp(0, 1)

        self.accuracy = CategoricalAccuracy()
        # self.metric = F1Measure(positive_label=1)
        self._token_embedder = text_field_embedder
        self._doc_encoder = doc_encoder
        self.ff = FeedForward(doc_encoder.get_output_dim(), num_layers=4,
                              hidden_dims=100,
                              activations=Activation.by_name("relu")())

        self.tag_projection_layer = Linear(self.ff.get_output_dim(), self.num_tags)
        self.choice_projection_layer = Linear(self.ff.get_output_dim(), 2)

    @overrides
    def forward(self,  # type: ignore
                graf: Dict[str, torch.LongTensor],
                const: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,  # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, seq_len, vocab_size)
        graf_emb = self._token_embedder(graf)
        graf_mask = util.get_text_field_mask(graf)

        # shape: (batch_size, vocab_size)
        graf_bow = self.bow_embedder(graf["tokens"]).clamp(0, 1)

        batch_size, _ = graf_bow.shape
        _, cm_dim = self.const_mat.shape

        # get similarity against all elements of the const mat
        # shape: (num_classes, vocab_size)
        batch_cm = self.const_mat.unsqueeze(0).expand(batch_size, self.num_tags, cm_dim).transpose(0,1)

        newcm = batch_cm.float() * graf_bow
        # shape: (batch, num_classes)
        bow_logits = newcm.transpose(1, 0).sum(-1)
        bow_logprob_logits = F.log_softmax(bow_logits, dim=1)

        ff = self.ff(self._doc_encoder(graf_emb, graf_mask))

        # shape: (batch, num_classes)
        logits = self.tag_projection_layer(ff)

        # shape: (batch, 2)
        choice_probs = F.softmax(self.choice_projection_layer(ff), dim=-1)

        projection_logprob_logits = F.log_softmax(logits, dim=-1)

        # shape: (batch, 2, num_classes)
        logits = torch.cat([bow_logprob_logits.unsqueeze(1), projection_logprob_logits.unsqueeze(1)], dim=1)
        logprob_logits = (choice_probs.unsqueeze(-1) * logits).sum(1)

        class_probabilities = torch.exp(logprob_logits)
        label_predictions = torch.argmax(logprob_logits, dim=-1)
        prediction_probs = torch.gather(class_probabilities, 1, label_predictions.unsqueeze(-1))

        output = {"prediction": label_predictions, "prediction_prob": prediction_probs}
        if label is not None:
            self.accuracy(logprob_logits, label)
            #self.metric(logprob_logits, label, torch.ones(1))
            logprob = torch.gather(logprob_logits, 1, label.unsqueeze(-1))

            loss = -logprob.sum()
            output["loss"] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        #prec, rec, f1 = self.metric.get_metric(reset=reset)

        return {"accuracy" : self.accuracy.get_metric(reset=reset)}
