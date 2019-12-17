# py-ctc-decode
Decoding output of CTC trained models.
Implementation was adapted from [https://github.com/PaddlePaddle/DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech), but in python for easier modifications. If you need the speed, use the Paddle implementation.
- Best Path
- Beam Search
- Beam Search with LM

## Usage
Following examples show how to use the different decoding strategies.
It is assumed that ``_`` is the blank symbol.
Probabilites are expected to be in natural log.

### Beam Search
```python
logits = [] # TxV
vocabulary = [' ', 'a', 'b', 'c', '_'] # V
decoder = ctcdecode.BestPathDecoder(
    vocabulary,
    num_workers=4
)

predictions = decoder.decode_batch(logits)
prediction = decoder.decode(logits[0])
```

### Beam Search
```python
logits = [] # TxV
vocabulary = [' ', 'a', 'b', 'c', '_'] # V
decoder = ctcdecode.BeamSearchDecoder(
    vocabulary,
    num_workers=4,
    beam_width=64,
    cutoff_prob=np.log(0.000001),
    cutoff_top_n=40
)

predictions = decoder.decode_batch(logits)
prediction = decoder.decode(logits[0])
```

### Beam Search with LM
```python
logits = [] # TxV
vocabulary = [' ', 'a', 'b', 'c', '_'] # V
alpha = 2.5 # LM Weight
beta = 0.0 # LM Usage Reward
word_lm_scorer = ctcdecode.WordKenLMScorer('path/to/kenlm', alpha, beta)
decoder = ctcdecode.BeamSearchDecoder(
    vocabulary,
    num_workers=4,
    beam_width=64,
    scorers=[word_lm_scorer],
    cutoff_prob=np.log(0.000001),
    cutoff_top_n=40
)

predictions = decoder.decode_batch(logits)
prediction = decoder.decode(logits[0])
```
