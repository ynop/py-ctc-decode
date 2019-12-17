import numpy as np

from ctcdecode.decoder import base


class BestPathDecoder(base.Decoder):

    def decode(self, probs):
        pred = []
        for t in np.argmax(probs, axis=1):
            c = self.vocab[t]
            if len(pred) == 0 or pred[-1] != c:
                pred.append(c)

        pred = ''.join(pred).replace('_', '')
        return pred
