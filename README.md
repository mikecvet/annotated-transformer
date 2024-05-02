# An Annotated Transformer

The best way to learn is by doing. This project is a from-scratch Transformer implementation, built using PyTorch with help from the following resources:
 - [Attention Is All you Need](https://arxiv.org/pdf/1706.03762)
 - [Stanford CS224N 2019 Readings](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/)
 - [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
 - [An even more annotated transformer](https://pi-tau.github.io/posts/transformer/)
 - [Building a Transformer with PyTorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)

This model comes with some starter training code, using the [Gigawords](https://huggingface.co/datasets/gigaword) dataset and the [uncased DistilBERT tokenizer](https://huggingface.co/distilbert/distilbert-base-uncased). It uses the following hyperparameter configuration:
 - Embeddings and hidden layer dimensionality: `256`
 - Inner feedforward layer dimensionality: `1024`
 - Number of attention heads: `8`
 - Number of encoder and decoder layers: `8`
 - Dropout rate: `0.1`
 - Label smoothing rate: `0.1`
 - Learning rate: `0.0001` (`AdamW`), weight decay: `0.05`
 - Gradient clipping norm: `2.0`
 - Temperature: `2.0`

The beam search implementation was copied from [mikecvet/beam](https://github.com/mikecvet/beam) and slightly modified

Printing an instance of the [Transformer](https://github.com/mikecvet/annotated-transformer/blob/main/src/transformer.py) class:

```
  Model: Transformer(src_vocab_size=32100, tgt_vocab_size=32100, dim=256, dim_inner=1024, heads=8, num_layers=4,
         max_seq_length=256, dropout_rate=Dropout(p=0.1, inplace=False), device=mps)
  	--> PositionalEncoder(dim=256, max_seq_len=256, device=mps)
  	--> Embedding(src_vocab_size=32100, dim=256) (encoder)
  	--> Embedding(tgt_vocab_size=32100, dim=256) (decoder)
  	--> [EncoderBlock(dim=256, num_attn_heads=8, num_layers=4, dim_inner=1024, dropout_rate=0.1, device=mps)]
  		--> (x4) [Encoder(dim=256, num_attn_heads=8, dim_inner=1024, dropout_rate=0.1, device=mps)]
  	--> [DecoderDecoder(dim=256, num_attn_heads=8, num_layers=4, dim_inner=1024, dropout_rate=0.1, device=mps)]
  		--> (x4) [Decoder(dim=256, num_attn_heads=8, dim_inner=1024, dropout_rate=0.1, device=mps)]
  	--> Linear(dim=256, tgt_vocab_size=32100)
  	--> Dropout(dropout_rate=0.1)
```

It was difficult to sufficiently train a model of this size on my Macbook Air, however the summarization task results are ~alright. I experimented with various configruations of attention heads, layers and dimensionality, and found that the above settings seemed to work best for this dataset specifically:

```
$ python3 src/main.py --temperature 2.0 --train 20
epoch loss: 2122.236578941345
Saved model data to transformer_state.data
epoch loss: 1960.4350719451904
Saved model data to transformer_state.data
epoch loss: 1945.9114050865173
Saved model data to transformer_state.data
epoch loss: 1935.8571014404297
(etc)

test struct: {
'document': 'the ruble fell to #,### here on friday from #,### on friday and the central bank intervened by selling ##.# million dollars , dealers said .',
'summary': 'ruble falls to #,### to the dollar',
'input_ids': tensor([  101,  1996, 14548,  2571,  3062,  2000,  1001,  1010,  1001,  1001,
         1001,  2182,  2006,  5958,  2013,  1001,  1010,  1001,  1001,  1001,
         2006,  5958,  1998,  1996,  2430,  2924, 21116,  2011,  4855,  1001,
         1001,  1012,  1001,  2454,  6363,  1010, 16743,  2056,  1012,   102,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     (etc)]),
'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (etc)]),
'labels': tensor([  101, 14548,  2571,  4212,  2000,  1001,  1010,  1001,  1001,  1001,
         2000,  1996,  7922,   102,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     (etc)])
}
labels: ['[CLS]', 'rub', '##le', 'falls', 'to', '#', ',', '#', '#', '#', 'to', 'the', 'dollar', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', (etc)]

(etc)

next step candidates:
	-9.81064919: [dollar rub french #, in falls to. at dollars million percent on - of billion euros january jobs]
	-9.81797765: [dollar rub french #, in falls to. at dollars million on percent march - euros billion of trade]
	-9.82840920: [dollar rub french #, in falls to. at dollars million percent on - of billion euros january]
	(more omitted)

generated sequence IDs: [101, 7922, 14548, 2413, 1001, 1010, 1999, 4212, 2000, 1012, 2012, 6363, 2454, 3867, 2006, 1011, 1997, 4551, 19329, 2254, 5841, 102]
tokens: ['[CLS]', 'dollar', 'rub', 'french', '#', ',', 'in', 'falls', 'to', '.', 'at', 'dollars', 'million', 'percent', 'on', '-', 'of', 'billion', 'euros', 'january', 'jobs', '[SEP]']
expected tokens (labels): ['[CLS]', 'rub', '##le', 'falls', 'to', '#', ',', '#', '#', '#', 'to', 'the', 'dollar', '[SEP]', '[PAD]', '[PAD]',
```
