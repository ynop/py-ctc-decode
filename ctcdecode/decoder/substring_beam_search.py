import numpy as np

from ctcdecode.prefix import State
from ctcdecode.decoder import beam_search


class SubstringBeamSearchDecoder(beam_search.BeamSearchDecoder):
    """
    Decoder using Beam-Search.
    In contrast to the default implementation,
    this creates new prefixes by adding substrings of symbols.

    e.g. Given Prefix is "hell" and a symbol "lo" comes in.
    A Prefix "hello" is created as well.
    """

    def __init__(self, vocab, num_workers=4, beam_width=64, scorers=None,
                 cutoff_prob=1.0, cutoff_top_n=40, only_repeating=True):
        super(SubstringBeamSearchDecoder, self).__init__(
            vocab, num_workers=num_workers, beam_width=beam_width,
            scorers=scorers, cutoff_prob=cutoff_prob, cutoff_top_n=cutoff_top_n
        )

        self.repeating = only_repeating

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
                        partial_symbols = [symbol]

                        for i in range(1, len(symbol) + 1):
                            if prefix.symbol is not None and \
                                    (not self.repeating or prefix.symbol.endswith(symbol[:i])):
                                partial_symbols.append(symbol[i:])

                        for partial_sym in partial_symbols:

                            # If the last symbol is repeated
                            # update the existing prefix
                            if partial_sym == prefix.symbol:
                                p = symbol_prob + prefix.p_non_blank_prev
                                prefix.add_p_non_blank(p)

                            new_prefix = prefixes.get_prefix(
                                prefix, partial_sym
                            )

                            if new_prefix is not None:
                                p = -np.inf

                                if partial_sym == prefix.symbol and \
                                        prefix.p_blank_prev > -np.inf:
                                    p = prefix.p_blank_prev + symbol_prob
                                elif prefix.symbol != partial_sym:
                                    p = prefix.score + symbol_prob

                                new_prefix.add_p_non_blank(p)

            prefixes.step()

        prefixes.finalize()

        return prefixes.best()
