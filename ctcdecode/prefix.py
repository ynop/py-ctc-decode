import numpy as np


class Prefix:
    """
    Class holding the state of a single Prefix/Beam.
    """

    __slots__ = [
        'value', 'symbol', 'p_blank', 'p_non_blank',
        'p_blank_prev', 'p_non_blank_prev', 'score', 'ext_weight'
    ]

    def __init__(self):
        self.value = ''
        self.symbol = None

        self.p_blank = -np.inf
        self.p_non_blank = -np.inf

        self.p_blank_prev = 0.0
        self.p_non_blank_prev = -np.inf

        self.score = 0.0

        self.ext_weight = 0.0

    def __repr__(self):
        return 'Prefix("{}", {}, "{}", {}, {})'.format(
            self.value, self.score, self.symbol,
            self.p_blank_prev,
            self.p_non_blank_prev
        )

    def step(self):
        self.p_blank_prev = self.p_blank
        self.p_non_blank_prev = self.ext_weight + self.p_non_blank

        self.score = np.logaddexp(self.p_blank_prev, self.p_non_blank_prev)

        self.p_blank = -np.inf
        self.p_non_blank = -np.inf

    def add_p_blank(self, p):
        self.p_blank = np.logaddexp(self.p_blank, p)

    def add_p_non_blank(self, p):
        self.p_non_blank = np.logaddexp(self.p_non_blank, p)


class State:
    """
    Class holding the state of the decoding process.
    """

    def __init__(self, size=64, scorers=None):
        self.prefixes = {
            '': Prefix()
        }
        self.step_prefixes = {}
        self.prev_prefixes = {}

        self.size = size
        self.scorers = scorers or []

    def __iter__(self):
        for p in list(self.prefixes.values()):
            yield p

    def get_prefix(self, prefix, symbol):
        new_value = prefix.value + symbol

        if new_value in self.prefixes.keys():
            return self.prefixes[new_value]
        elif new_value in self.prev_prefixes.keys():
            new_prefix = self.prev_prefixes[new_value]
            self.step_prefixes[new_value] = new_prefix
            return new_prefix
        else:
            new_prefix = Prefix()
            new_prefix.value = new_value

            new_prefix.p_blank_prev = prefix.p_blank_prev
            new_prefix.p_non_blank_prev = prefix.p_non_blank_prev
            new_prefix.score = prefix.score

            new_prefix.p_blank = -np.inf
            new_prefix.p_non_blank = -np.inf

            new_prefix.symbol = symbol
            new_prefix.ext_weight = 0.0

            for scorer in self.scorers:
                if not scorer.is_valid_prefix(new_value):
                    return None

                new_prefix.ext_weight += scorer.score_prefix(new_prefix)

            self.step_prefixes[new_value] = new_prefix
            return new_prefix

        return None

    def step(self):
        self.prefixes.update(self.step_prefixes)
        self.step_prefixes = {}

        for prefix in self.prefixes.values():
            prefix.step()

        p_sorted = sorted(
            self.prefixes.items(),
            key=lambda x: x[1].score,
            reverse=True
        )

        self.prefixes = {}
        for value, prefix in p_sorted[:self.size]:
            self.prefixes[value] = prefix

        self.prev_prefixes = {}
        for value, prefix in p_sorted[self.size:]:
            self.prev_prefixes[value] = prefix

    def finalize(self):
        for scorer in self.scorers:
            for prefix in self:
                ext_score = scorer.final_prefix_score(prefix)
                prefix.score += ext_score

    def best(self):
        p_sorted = sorted(
            self.prefixes.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        return p_sorted[0][0]
