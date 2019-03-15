from typing import Dict, List, Any, Optional

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from overrides import overrides
from allennlp.models.model import Model
import torch
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from torch.nn import Parameter
from torch.nn.modules.linear import Linear
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder, TimeDistributed
import allennlp.nn.util as util
from allennlp.nn import Activation
import torch.nn.functional as F
from allennlp.data import Vocabulary

from mylib.hamming_loss import HammingLoss
from mylib.json2lines import JsonConverter
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer


@Model.register("legal_classifier")
class LegalClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 doc_encoder: Seq2VecEncoder,
                 const_path: str,
                 tokens_namespace: str,
                 use_sim: bool = True,
                 use_classifier: bool = True,
                 ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self.num_tags = vocab.get_vocab_size("labels")

        self._token_embedder = text_field_embedder
        self._doc_encoder = doc_encoder

        if not use_sim:
            raise Exception("use_sim option is false, but it must be true for this to work")

        if use_classifier:
            print("Warning: use_classifier option does nothing now...")

        self.use_sim = use_sim
        self.use_classifier = use_classifier

        # I actually want to use the one from the config, but not sure how to do that.
        _spacy_word_splitter = SpacyWordSplitter()
        token_indexer = PretrainedBertIndexer("bert-base-cased", do_lowercase=False, use_starting_offsets=True)

        jc = JsonConverter()
        const, links = jc._read_const(const_path)

        # the extra 1 is for the "unmatched" label.
        print(vocab.get_token_to_index_vocabulary("labels"))
        print(const.keys())
        assert self.num_tags == len(const) + 1, "Num tags ({}) doesn't match the size of the constitution+1 ({})".format(self.num_tags, len(const) + 1)

        # this will be the threshold that chooses
        self.threshold = Parameter(torch.Tensor([0.5]))

        if self.use_sim:
            # create the constitution matrix. Every element is one of the groups.
            tagmap = self.vocab.get_index_to_token_vocabulary("labels")
            self.const_dict = {}
            indices = []
            for i in range(self.num_tags):
                tagname = tagmap[i]
                if tagname != "unmatched":
                    const_text = const[tagname]
                else:
                    const_text = "@@pad@@"

                const_toks = _spacy_word_splitter.split_words(const_text)
                # truncate so BERT is happy.
                const_toks = const_toks[:250]
                const_indices = token_indexer.tokens_to_indices(const_toks, vocab, tokens_namespace)
                indices.append(const_indices)

            max_len = max(map(lambda j: len(j[tokens_namespace]), indices))
            max_offset_len = max(map(lambda j: len(j["tokens-offsets"]), indices))

            const_tensor = torch.zeros(self.num_tags, max_len).long()
            const_tensor_offsets = torch.zeros(self.num_tags, max_offset_len).long()
            const_tensor_mask = torch.zeros(self.num_tags, max_offset_len).long()
            for i, ind in enumerate(indices):
                toks = ind[tokens_namespace]
                mask = ind["mask"]
                const_tensor[i, :len(toks)] = torch.LongTensor(toks)
                const_tensor_offsets[i, :len(ind["tokens-offsets"])] = torch.LongTensor(ind["tokens-offsets"])
                const_tensor_mask[i, :len(mask)] = torch.LongTensor(mask)

            const_tokens = {tokens_namespace: const_tensor, "tokens-offsets": const_tensor_offsets, "mask": const_tensor_mask}

            print("Embedding the constitution... this could take a minute...")
            self.const_mask = util.get_text_field_mask(const_tokens)
            self.const_emb = self._token_embedder(const_tokens).detach()
            print("Done embedding the constitution.")

            if torch.cuda.is_available():
                self.const_emb = self.const_emb.cuda()
                self.const_mask = self.const_mask.cuda()

        self.hamming = HammingLoss()
        # self.metric = F1Measure(positive_label=1)

        self.ff = FeedForward(doc_encoder.get_output_dim(), num_layers=4,
                              hidden_dims=100,
                              activations=Activation.by_name("relu")())

        #self.tag_projection_layer = Linear(self.ff.get_output_dim(), self.num_tags)
        #self.choice_projection_layer = Linear(self.ff.get_output_dim(), 2)

        self.sim_ff = TimeDistributed(FeedForward(doc_encoder.get_output_dim(), num_layers=1,
                                                  hidden_dims=2,
                                                  activations=Activation.by_name("relu")()))

    @overrides
    def forward(self,  # type: ignore
                graf: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,  # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, seq_len, vocab_size)
        graf_emb = self._token_embedder(graf)
        graf_mask = util.get_text_field_mask(graf)
        graf_doc_emb = self._doc_encoder(graf_emb, graf_mask)

        #if self.use_sim:
        const_doc_emb = self._doc_encoder(self.const_emb, self.const_mask)

        batch_size, _, _ = graf_emb.shape
        _, cm_dim = const_doc_emb.shape

        # get similarity against all elements of the const mat
        # shape: (num_classes, batch_size, doc_encoder)
        batch_cm = const_doc_emb.unsqueeze(0).expand(batch_size, self.num_tags, cm_dim).transpose(0, 1)

        # shape (batch, num_classes, doc_encoder)
        newcm = (batch_cm.float() * graf_doc_emb).transpose(1, 0)

        # I want something with the shape: (batch, num_classes, 2)
        # where the last dimension is yes/no
        decisions = self.sim_ff(newcm)
        decisions_logprob = torch.log_softmax(decisions, dim=2)
        # shape: (batch, num_classes)
        label_predictions = torch.argmax(decisions, dim=2)

        # this means that we are just taking a dot product
        # shape: (batch, num_classes)
        #bow_logits = newcm.sum(dim=-1)

        # shape: (batch, num_classes)
        # bow_logits = self.sim_ff(newcm).squeeze(-1)
        #bow_logprob_logits = F.log_softmax(bow_logits, dim=1)

        # if self.use_classifier:
        #     # shape: (batch, num_classes)
        #     ff = self.ff(graf_doc_emb)
        #     logits = self.tag_projection_layer(ff)
        #     projection_logprob_logits = F.log_softmax(logits, dim=-1)

        # if self.use_sim and self.use_classifier:
        #     # shape: (batch, 2)
        #     choice_probs = F.softmax(self.choice_projection_layer(ff), dim=-1)
        #
        #     # shape: (batch, 2, num_classes)
        #     logits = torch.cat([bow_logprob_logits.unsqueeze(1), projection_logprob_logits.unsqueeze(1)], dim=1)
        #     logprob_logits = (choice_probs.unsqueeze(-1) * logits).sum(1)
        #if self.use_sim:
        #logprob_logits = bow_logprob_logits

        # elif self.use_classifier:
        #     logprob_logits = projection_logprob_logits
        #     choice_probs = "none"

        class_probabilities = torch.exp(decisions_logprob)
        #label_predictions = class_probabilities > self.threshold

        output = {"prediction": label_predictions, "class_probabilities" : class_probabilities}
        if label is not None:
            #print(label_predictions)
            self.hamming(label_predictions, label)

            invlabel = 1-label.float()
            # something to help with the label imbalance...
            invlabel = invlabel / self.num_tags
            # shape: (batch, num_classes, 2)
            newlabel = torch.stack([invlabel, label.float()], dim=-1)

            #print(newlabel)
            #print(decisions_logprob)

            logprob = decisions_logprob * newlabel.float()

            #print(logprob)

            loss = -logprob.sum()
            output["loss"] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        #prec, rec, f1 = self.metric.get_metric(reset=reset)

        return {"accuracy": self.hamming.get_metric(reset=reset)}
