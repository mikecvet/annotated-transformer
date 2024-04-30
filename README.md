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

