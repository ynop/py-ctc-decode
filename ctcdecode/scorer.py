import collections
import uuid

import numpy as np
import kenlm

OOV_WORD_PENALTY = -1000.0


class Scorer:
    """
    Base class for a external scorer.
    This can be used to integrate for example a language model.
    """

    def score_prefix(self, prefix):
        """
        Return a score (log base e) for the given prefix.
        """
        pass

    def final_prefix_score(self, prefix):
        """
        Return a score (log base e) for the given prefix,
        considering the prefix won't be extended anymore.
        This is called for every prefix at the end,
        whether ``score_prefix`` was already called or not.
        """
        pass

    def is_valid_prefix(self, value):
        """
        Return ``True``, if the given prefix is valid.
        """
        pass


class WordKenLMScorer(Scorer):

    def __init__(self, path, alpha, beta):
        self.path = path
        self.alpha = alpha
        self.beta = beta

        self.lm = kenlm.Model(path)

        self.words = self._get_words(path)
        self.word_prefixes = self._get_word_prefixes(self.words)

        self.idx = uuid.uuid1()

    def score_prefix(self, prefix):
        if prefix.symbol == ' ':
            words = prefix.value.strip().split(' ')
            cond_prob = self.get_cond_log_prob(words)
            cond_prob *= self.alpha
            cond_prob += self.beta

            return self._to_base_e(cond_prob)

        return 0.0

    def final_prefix_score(self, prefix):
        if prefix.symbol != ' ':
            words = prefix.value.strip().split(' ')
            cond_prob = self.get_cond_log_prob(words)
            cond_prob *= self.alpha
            cond_prob += self.beta

            return self._to_base_e(cond_prob)

        return 0.0

    def is_valid_prefix(self, value):
        last_word = value.strip().split(' ')[-1]
        return last_word in self.word_prefixes[len(last_word)]

    def get_cond_log_prob(self, sequence):
        sequence = sequence[-self.lm.order:]

        in_state = kenlm.State()
        self.lm.NullContextWrite(in_state)
        out_state = kenlm.State()

        for word in sequence:
            if word not in self.words:
                return OOV_WORD_PENALTY

            lm_prob = self.lm.BaseScore(
                in_state, word, out_state
            )
            tmp_state = in_state
            in_state = out_state
            out_state = tmp_state

        return lm_prob

    def _to_base_e(self, x):
        return x / np.log(10)

    def _get_words(self, path):
        words = set()

        with open(path, 'r') as f:
            start_1_gram = False
            end_1_gram = False

            while not end_1_gram:
                line = f.readline().strip()

                if line == '\\1-grams:':
                    print('found 1gram')
                    start_1_gram = True

                elif line == '\\2-grams:':
                    print('found 2gram')
                    end_1_gram = True

                elif start_1_gram and line != '':
                    parts = line.split('\t')
                    if len(parts) == 3:
                        words.add(parts[1])

        return words

    def _get_word_prefixes(self, words):
        word_prefixes = collections.defaultdict(set)

        for word in words:
            for i in range(1, len(word) + 1):
                word_prefixes[i].add(word[:i])

        return word_prefixes


class CharOfWordKenLMScorer(Scorer):

    def __init__(self, path, alpha, beta):
        self.path = path
        self.alpha = alpha
        self.beta = beta

        self.lm = kenlm.Model(path)

    def score_prefix(self, prefix):
        if prefix.symbol != ' ':
            words = prefix.value.strip().split(' ')
            last_word = words[-1]

            total_cond_prob = 0.0

            # Account for multi-char symbols
            for i in range(len(prefix.symbol)):
                part = last_word[:len(last_word)-i]
                chars = list(part)

                cond_prob = self.get_cond_log_prob(chars)
                total_cond_prob += cond_prob * self.alpha
                total_cond_prob += self.beta

            return self._to_base_e(total_cond_prob)

        return 0.0

    def final_prefix_score(self, prefix):
        return 0.0

    def is_valid_prefix(self, value):
        return True

    def get_cond_log_prob(self, sequence):
        sequence = sequence[-self.lm.order:]

        in_state = kenlm.State()
        self.lm.NullContextWrite(in_state)
        out_state = kenlm.State()

        for word in sequence:
            lm_prob = self.lm.BaseScore(
                in_state, word, out_state
            )
            tmp_state = in_state
            in_state = out_state
            out_state = tmp_state

        return lm_prob

    def _to_base_e(self, x):
        return x / np.log(10)
