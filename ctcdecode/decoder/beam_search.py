import numpy as np

from ctcdecode.prefix import State
from ctcdecode.decoder import base


class BeamSearchDecoder(base.Decoder):

    def __init__(self, vocab, num_workers=4, beam_width=64, scorers=None,
                 cutoff_prob=1.0, cutoff_top_n=40):
        super(BeamSearchDecoder, self).__init__(vocab, num_workers=num_workers)

        self.beam_width = beam_width
        self.scorers = scorers
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n

    def decode(self, probs):
        # Num time steps
        nT = probs.shape[0]

        # Initialize prefixes
        prefixes = State(
            scorers=self.scorers,
            size=self.beam_width
        )

        # Iterate over timesteps
        for t in range(nT):
            step_probs = probs[t]
            pruned_step_probs = self._get_pruned_vocab_indices(step_probs)

            # Iterate over symbols
            for v in pruned_step_probs:
                symbol = self.vocab[v]
                symbol_prob = step_probs[v]

                # Iterate over prefixes
                for prefix in prefixes:

                    # If there is a blank, we extend the existing prefix
                    if symbol == '_':
                        prefix.add_p_blank(symbol_prob + prefix.score)

                    else:

                        # If the last symbol is repeated
                        # update the existing prefix
                        if symbol == prefix.symbol:
                            p = symbol_prob + prefix.p_non_blank_prev
                            prefix.add_p_non_blank(p)

                        new_prefix = prefixes.get_prefix(prefix, symbol)

                        if new_prefix is not None:
                            p = -np.inf

                            if symbol == prefix.symbol and \
                                    prefix.p_blank_prev > -np.inf:
                                p = prefix.p_blank_prev + symbol_prob

                            elif prefix.symbol != symbol:
                                p = prefix.score + symbol_prob

                            new_prefix.add_p_non_blank(p)

            prefixes.step()

        prefixes.finalize()

        return prefixes.best()

    def _get_pruned_vocab_indices(self, log_probs):
        """ Return vocab indices of pruned probabilities of a time step. """

        index_to_prob = [(k, log_probs[k]) for k in range(log_probs.shape[0])]
        index_to_prob = sorted(index_to_prob, key=lambda x: x[1], reverse=True)

        if self.cutoff_top_n < len(index_to_prob):
            index_to_prob = index_to_prob[:self.cutoff_top_n]

        if self.cutoff_prob < 1.0:
            filtered = []
            for x in index_to_prob:
                if x[1] >= self.cutoff_prob:
                    filtered.append(x)
            index_to_prob = filtered

        return [x[0] for x in index_to_prob]
