import numpy as np
from itertools import chain
from enum import IntEnum


def _get_row_ids(ar):
    row_ids = ar[:, 0].copy()
    steps = ar[:, :-1].max(axis=0) + 1
    for i in range(1, ar.shape[1]):
        row_ids += ar[:, i] * np.prod(steps[:i])
    return row_ids


def _distribution(ar):
    nans = np.isnan(ar)

    if ar.ndim == 1:
        if nans.any():
            ar = ar[~nans]
    else:
        if nans.any():
            ar = ar[~nans.any(axis=1)]
        # Calling `np.unique` with `axis=0` works, but it slows down the main thread.
        # The implementation probably doesn't release GIL. Until it gets fixed use
        # `_get_row_ids` (should work, assuming `ar` contains non-negative integers).
        ar = _get_row_ids(ar)
    _, counts = np.unique(ar, return_counts=True)
    return counts / ar.shape[0]


def _entropy(ar):
    p = _distribution(ar)
    return -np.sum(p * np.log2(p))


class InteractionScorer:
    def __init__(self, data):
        self.data = data
        self.class_entropy = 0
        self.information_gain = np.zeros(data.X.shape[1])

        # Precompute information gain of each attribute for faster overall
        # computation and to create heuristic. Only removes necessary NaN values
        # to keep as much data as possible and keep entropies and information gains
        # invariant of third attribute.
        # In certain situations this can cause unexpected results i.e. negative
        # information gains or negative interactions lower than individual
        # attribute information.
        self._precompute()

    def _precompute(self):
        self.class_entropy = _entropy(self.data.Y)
        for attr in range(self.information_gain.size):
            self.information_gain[attr] = self.class_entropy \
                                          + _entropy(self.data.X[:, attr]) \
                                          - _entropy(np.column_stack((self.data.X[:, attr], self.data.Y)))

    def __call__(self, attr1, attr2):
        attrs = np.column_stack((self.data.X[:, attr1], self.data.X[:, attr2]))
        return self.class_entropy \
               - self.information_gain[attr1] \
               - self.information_gain[attr2] \
               + _entropy(attrs) \
               - _entropy(np.column_stack((attrs, self.data.Y)))

    def normalize(self, score):
        return score / self.class_entropy


class Heuristic:
    def __init__(self, weights, heuristic_type=None):
        self.n_attributes = len(weights)
        self.attributes = np.arange(self.n_attributes)
        if heuristic_type == HeuristicType.INFO_GAIN:
            self.attributes = self.attributes[np.argsort(weights)]
        else:
            np.random.shuffle(self.attributes)

    def generate_states(self):
        # prioritize two mid ranked attributes over highest first
        for s in range(1, self.n_attributes * (self.n_attributes - 1) // 2):
            for i in range(max(s - self.n_attributes + 1, 0), (s + 1) // 2):
                yield self.attributes[i], self.attributes[s - i]

    def get_states(self, initial_state):
        states = self.generate_states()
        if initial_state is not None:
            while next(states) != initial_state:
                pass
            return chain([initial_state], states)
        return states


class HeuristicType(IntEnum):
    RANDOM, INFO_GAIN = 0, 1

    @staticmethod
    def items():
        return ["Random Search", "Information Gain Heuristic"]
