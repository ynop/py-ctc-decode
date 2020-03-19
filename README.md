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

### Best Path
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
logits = [
    [-1.1906, -1.0623, -1.7766, -1.7086],
    [-1.4091, -1.4424, -1.1923, -1.5336],
    [-1.4091, -1.6900, -1.6956, -0.9477],
    [-1.3715, -1.2527, -1.7445, -1.2524],
    [-1.2577, -1.2588, -1.3380, -1.7759]
] # 5x4 TimeSteps x Softmax over Vocabulary (NATURAL LOG !!!)
vocabulary = [' ', 'a', 'b', '_'] # 4
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

prediction = decoder.decode(logits) # text (e.g. "a b")

# Batch decoding for multiple utterances
batch = [logits, ....]
predictions = decoder.decode_batch(batch)
```
