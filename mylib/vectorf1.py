from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("vectorf1")
class VectorF1(Metric):
    """
    """
    def __init__(self, unmatched_index: int) -> None:
        self.unmatched_index = unmatched_index
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ..., num_classes).
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        #predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        if len(gold_labels.shape) != 2:
            gold_labels = gold_labels.unsqueeze(0)
            predictions = predictions.unsqueeze(0)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.shape != predictions.shape:
            raise ConfigurationError("gold_labels must have the same shape as predictions.size() "
                                     "found tensor of shape: {}".format(predictions.size()))

        gold_labels[:, self.unmatched_index] = 0
        predictions[:, self.unmatched_index] = 0

        self._true_positives += ((gold_labels+predictions) == 2).sum()
        self._true_negatives += ((gold_labels+predictions) == 0).sum()
        self._false_negatives += ((gold_labels - predictions) == 1).sum()
        self._false_positives += ((gold_labels - predictions) == -1).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated F1.
        """

        fp = float(self._false_positives)
        tp = float(self._true_positives)
        fn = float(self._false_negatives)
        eps = 1e-10
        # print("eps", eps)
        # print("denom", fp + tp + eps)
        # print("tp", tp)
        # print("fp", fp)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_measure = 2. * ((precision * recall) / (precision + recall + eps))
        if reset:
            self.reset()
        return precision, recall, f1_measure


    @overrides
    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
