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
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder, TimeDistributed
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

        self._token_embedder = text_field_embedder
        self._doc_encoder = doc_encoder

        # I actually want to use the one from the config, but not sure how to do that.
        _spacy_word_splitter = SpacyWordSplitter()
        token_indexer = SingleIdTokenIndexer(namespace="tokens", lowercase_tokens=True,
                                             end_tokens=["@@pad@@", "@@pad@@", "@@pad@@", "@@pad@@"])
        # TODO: turn this into an argument
        # self.bow_embedder = BagOfWordCountsTokenEmbedder(vocab, "tokens")

        jc = JsonConverter()
        const, links = jc._read_const(const_path)

        # the extra 1 is for the "unmatched" label.
        assert self.num_tags == len(const) + 1

        # create the constitution matrix. Every element is one of the groups.
        tagmap = self.vocab.get_index_to_token_vocabulary("labels")
        print(tagmap)
        self.const_dict = {}
        indices = []
        for i in range(self.num_tags):
            tagname = tagmap[i]
            if tagname != "unmatched":
                const_text = const[tagname]
            else:
                const_text = "@@pad@@"

            const_toks = _spacy_word_splitter.split_words(const_text)
            const_indices = token_indexer.tokens_to_indices(const_toks, vocab, "tokens")
            indices.append(const_indices)

        max_len = max(map(lambda j: len(j["tokens"]), indices))

        const_tensor = torch.zeros(self.num_tags, max_len).long()
        for i, ind in enumerate(indices):
            toks = ind["tokens"]
            const_tensor[i, :len(toks)] = torch.LongTensor(toks)

        self.const_tokens = {"tokens": const_tensor}

        self.accuracy = CategoricalAccuracy()
        # self.metric = F1Measure(positive_label=1)

        self.ff = FeedForward(doc_encoder.get_output_dim(), num_layers=4,
                              hidden_dims=100,
                              activations=Activation.by_name("relu")())

        self.tag_projection_layer = Linear(self.ff.get_output_dim(), self.num_tags)
        self.choice_projection_layer = Linear(self.ff.get_output_dim(), 2)

        self.sim_ff = TimeDistributed(FeedForward(self._doc_encoder.get_output_dim(), num_layers=1,
                                                  hidden_dims=1,
                                                  activations=Activation.by_name("relu")()))

    @overrides
    def forward(self,  # type: ignore
                graf: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,  # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        #const_mat = torch.FloatTensor(self.num_tags, self._doc_encoder.get_output_dim())
        #if torch.cuda.is_available():
        #    const_mat = const_mat.cuda()

        const_mask = util.get_text_field_mask(self.const_tokens)
        const_emb = self._token_embedder(self.const_tokens)
        const_doc_emb = self._doc_encoder(const_emb, const_mask)

        # for i in self.const_dict:
        #     t, tm = self.const_dict[i]
        #     const_embs = self._doc_encoder(self._token_embedder(t), tm)
        #     const_mat[i, :] = const_embs

        # shape: (batch_size, seq_len, vocab_size)
        graf_emb = self._token_embedder(graf)
        graf_mask = util.get_text_field_mask(graf)

        graf_doc_emb = self._doc_encoder(graf_emb, graf_mask)

        batch_size, _, _ = graf_emb.shape
        _, cm_dim = const_doc_emb.shape

        # get similarity against all elements of the const mat
        # shape: (num_classes, batch_size, vocab_size)
        batch_cm = const_doc_emb.unsqueeze(0).expand(batch_size, self.num_tags, cm_dim).transpose(0, 1)

        # shape (batch, num_classes, vocab_size)
        newcm = (batch_cm.float() * graf_doc_emb).transpose(1, 0)
        # shape (batch, vocab_size, num_classes)

        # this means that we are just taking a dot product
        # shape: (batch, num_classes)
        bow_logits = newcm.sum(dim=-1)

        # shape: (batch, num_classes)
        # bow_logits = self.sim_ff(newcm).squeeze(-1)
        bow_logprob_logits = F.log_softmax(bow_logits, dim=1)

        # FIXME: break these into variables!!
        use_sim = True
        use_classifier = True

        # shape: (batch, num_classes)
        ff = self.ff(graf_doc_emb)
        logits = self.tag_projection_layer(ff)

        # shape: (batch, 2)
        choice_probs = F.softmax(self.choice_projection_layer(ff), dim=-1)

        projection_logprob_logits = F.log_softmax(logits, dim=-1)

        if use_sim and use_classifier:
            # shape: (batch, 2, num_classes)
            logits = torch.cat([bow_logprob_logits.unsqueeze(1), projection_logprob_logits.unsqueeze(1)], dim=1)
            logprob_logits = (choice_probs.unsqueeze(-1) * logits).sum(1)
        elif use_sim:
            logprob_logits = bow_logprob_logits
        elif use_classifier:
            logprob_logits = projection_logprob_logits

        class_probabilities = torch.exp(logprob_logits)
        label_predictions = torch.argmax(logprob_logits, dim=-1)
        prediction_probs = torch.gather(class_probabilities, 1, label_predictions.unsqueeze(-1))

        output = {"prediction": label_predictions, "prediction_prob": prediction_probs, "choice_prob": choice_probs}
        if label is not None:
            self.accuracy(logprob_logits, label)
            logprob = torch.gather(logprob_logits, 1, label.unsqueeze(-1))

            loss = -logprob.sum()
            output["loss"] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        #prec, rec, f1 = self.metric.get_metric(reset=reset)

        return {"accuracy": self.accuracy.get_metric(reset=reset)}
