from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("vectorf1")
class VectorF1(Metric):
    """
    """
    def __init__(self) -> None:
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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.shape != predictions.shape:
            raise ConfigurationError("gold_labels must have the same shape as predictions.size() "
                                     "found tensor of shape: {}".format(predictions.size()))

        #predictions = predictions.view(-1, num_classes)
        #gold_labels = gold_labels.view(-1, num_classes)

        # input has size (batch, num_classes)

        # this will do nothing...

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
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure


    @overrides
    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
