import numpy as np
from itertools import chain
from enum import IntEnum


class Interaction:
    def __init__(self, disc_data):
        self.data = disc_data
        self.n_attrs = len(self.data.domain.attributes)
        self.class_h = self.entropy(self.data.Y)
        self.attr_h = np.zeros(self.n_attrs)
        self.gains = np.zeros(self.n_attrs)
        self.removed_h = np.zeros((self.n_attrs, self.n_attrs))

        # Precompute information gain of each attribute for faster overall
        # computation and to create heuristic. Only removes necessary NaN values
        # to keep as much data as possible and keep entropies and information gains
        # invariant of third attribute.
        # In certain situations this can cause unexpected results i.e. negative
        # information gains or negative interactions lower than individual
        # attribute information.
        self.compute_gains()

    @staticmethod
    def distribution(ar):
        nans = np.isnan(ar)
        if nans.any():
            if ar.ndim == 1:
                ar = ar[~nans]
            else:
                ar = ar[~nans.any(axis=1)]
        _, counts = np.unique(ar, return_counts=True, axis=0)
        return counts / len(ar)

    def entropy(self, ar):
        p = self.distribution(ar)
        return -np.sum(p * np.log2(p))

    def compute_gains(self):
        for attr in range(self.n_attrs):
            self.attr_h[attr] = self.entropy(self.data.X[:, attr])
            self.gains[attr] = self.attr_h[attr] + self.class_h \
                               - self.entropy(np.c_[self.data.X[:, attr], self.data.Y])

    def __call__(self, attr1, attr2):
        attrs = np.c_[self.data.X[:, attr1], self.data.X[:, attr2]]
        self.removed_h[attr1, attr2] = self.entropy(attrs) + self.class_h - self.entropy(np.c_[attrs, self.data.Y])
        score = self.removed_h[attr1, attr2] - self.gains[attr1] - self.gains[attr2]
        return score


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
