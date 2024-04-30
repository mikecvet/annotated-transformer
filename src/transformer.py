from decoder_block import DecoderBlock
from encoder_block import EncoderBlock
from pos_encoder import PositionalEncoding
import torch
import torch.nn as nn

class Transformer(nn.Module):
  """
  Implements a Transformer model for sequence-to-sequence tasks. This class combines the Transformer encoder and decoder 
  layers, embedding layers for input and output sequences, and a final linear layer to project decoder output to the target 
  vocabulary size.
  """

  def __init__(
      self, 
      src_vocab_size, 
      tgt_vocab_size, 
      dim, 
      dim_inner, 
      num_attn_heads, 
      num_layers, 
      max_seq_len, 
      dropout_rate,
      device):
    """
    Initializes a new Transformer model.

    Attributes:
      encoder_embedding (nn.Embedding):        Embedding layer for the source sequence vocabulary.
      decoder_embedding (nn.Embedding):        Embedding layer for the target sequence vocabulary.
      positional_encoding (PositionalEncoder): Adds positional information to the embeddings.
      encoder_layers (nn.ModuleList):          A list of Encoder layers.
      decoder_layers (nn.ModuleList):          A list of Decoder layers.
      linear (nn.Linear):                      A linear layer that projects the decoder's output to the target vocabulary size.
      dropout (nn.Dropout):                    Dropout layer applied to the embeddings.
      device: (torch.Device):                  The device used for computation

    Parameters:
      src_vocab_size (int):   Size of the source vocabulary.
      tgt_vocab_size (int):   Size of the target vocabulary.
      dim (int):              Dimensionality of the embeddings and hidden layers.
      dim_inner (int):        Dimensionality of the inner feedforward layers.
      num_attn_heads (int):   Number of attention heads in the multi-head attention mechanisms.
      num_layers (int):       Number of encoder and decoder layers.
      max_seq_len (int):      Maximum sequence length for positional encodings.
      dropout_rate (float):   Dropout rate for embedding layers.
      device: (torch.Device): The device to be used for computation

    Model architecture:
    
    Transformer 
      --> PositionalEncoder
      --> Encoder Embedding
      --> Decoder Embedding
      --> [Encoder] (x num_layers)
      |     --> MultiHeadAttention
      |     |     --> Linear (x4)
      |     --> PositionWiseFeedForward
      |     |    --> Linear (x2)
      |     |    --> ReLU
      |     --> LayerNorm (x2)
      |     --> Dropout
      --> [Decoder] (x num_layers)
      |     --> MultiHeadAttention (x2)
      |     |    --> Linear (x4)
      |     --> PositionWiseFeedForward
      |     |    --> Linear (x2)
      |     |    --> ReLU
      |     --> LayerNorm (x3)
      |     --> Dropout
      --> Linear
      --> Dropout
    """  
    super(Transformer, self).__init__()

    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.dimensionality = dim
    self.dimensionality_inner = dim_inner
    self.num_attn_heads = num_attn_heads
    self.num_layers = num_layers
    self.max_seq_len = max_seq_len
    self.dropout_rate = dropout_rate
    self.device = device

    self.positional_encoding = PositionalEncoding(dim, max_seq_len, device)

    # https://arxiv.org/pdf/1706.03762.pdf # 3.4
    self.encoder_embedding = nn.Embedding(src_vocab_size, dim).to(device)
    self.decoder_embedding = nn.Embedding(tgt_vocab_size, dim).to(device)

    self.encoder_block = EncoderBlock(dim, num_attn_heads, num_layers, dim_inner, dropout_rate, device)
    self.decoder_block = DecoderBlock(dim, num_attn_heads, num_layers, dim_inner, dropout_rate, device)

    self.linear = nn.Linear(dim, tgt_vocab_size)
    self.dropout = nn.Dropout(dropout_rate)

  def __str__(self):
    S = ""
    S += f"Transformer(src_vocab_size={self.src_vocab_size}, tgt_vocab_size={self.tgt_vocab_size}, dim={self.dimensionality}, "
    S += f"dim_inner={self.dimensionality_inner}, heads={self.num_attn_heads}, num_layers={self.num_layers}, max_seq_length={self.max_seq_len}, "
    S += f"dropout_rate={self.dropout}, device={self.device})\n"

    S += f"\t--> PositionalEncoder(dim={self.dimensionality}, max_seq_len={self.max_seq_len}, device={self.device})\n"
    S += f"\t--> Embedding(src_vocab_size={self.src_vocab_size}, dim={self.dimensionality}) (encoder)\n"
    S += f"\t--> Embedding(tgt_vocab_size={self.tgt_vocab_size}, dim={self.dimensionality}) (decoder)\n"

    S += f"\t--> [EncoderBlock(dim={self.dimensionality}, num_attn_heads={self.num_attn_heads}, num_layers={self.num_layers}, dim_inner={self.dimensionality_inner}, dropout_rate={self.dropout_rate}, device={self.device})]\n"
    S += f"\t\t--> (x{self.num_layers}) [Encoder(dim={self.dimensionality}, num_attn_heads={self.num_attn_heads}, dim_inner={self.dimensionality_inner}, dropout_rate={self.dropout_rate}, device={self.device})]\n"

    S += f"\t--> [DecoderDecoder(dim={self.dimensionality}, num_attn_heads={self.num_attn_heads}, num_layers={self.num_layers}, dim_inner={self.dimensionality_inner}, dropout_rate={self.dropout_rate}, device={self.device})]\n"
    S += f"\t\t--> (x{self.num_layers}) [Decoder(dim={self.dimensionality}, num_attn_heads={self.num_attn_heads}, dim_inner={self.dimensionality_inner}, dropout_rate={self.dropout_rate}, device={self.device})]\n"

    S += f"\t--> Linear(dim={self.dimensionality}, tgt_vocab_size={self.tgt_vocab_size})\n"
    S += f"\t--> Dropout(dropout_rate={self.dropout_rate})\n"

    return S

  def generate_mask(self, src, tgt):
    """
    Generates masks for the source and target sequences. The source mask hides padding tokens in the source sequence.
    The target mask additionally prevents attention to future tokens in the target sequence, ensuring that predictions 
    for position `i` can depend only on known outputs at positions less than `i`.

    Parameters:
      src (Tensor): Source sequence tensor
      tgt (Tensor): Target sequence tensor

    Returns:
      Tuple[Tensor, Tensor]: The source and target masks
    """

    # Look for [PAD] tokens which have an ID of 0
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    if tgt is None:
      # If tgt is None (during the first step of free-running mode), create an empty mask
      tgt_mask = torch.zeros(1, 1, src.size(-1)).to(self.device).bool()
    else:
      tgt_mask = (tgt != 0).unsqueeze(1)
      if tgt.ndim > 1:
        # For training and subsequent steps of free-running mode with multiple tokens
        tgt_mask = tgt_mask.unsqueeze(3)
        seq_length = tgt.size(1)
        no_peek_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1).to(self.device)).bool()
        tgt_mask = tgt_mask & no_peek_mask
      else:
        # For the first step of free-running mode with a single token
        tgt_mask = tgt_mask.unsqueeze(2)

    return src_mask, tgt_mask

  def encode(self, src, mask):
    # Embed the source sequence through the encoder embedding layer.
    # Then apply positional encoding to inject sequence position information, and finally dropout
    z = self.encoder_embedding(src)
    z = self.positional_encoding(z)
    z = self.dropout(z)

    # Pass the encoder output through each encoder layer in the encoder block
    # Each layer applies self-attention, followed by a feedforward network, with normalization and dropout applied
    z = self.encoder_block(z, mask)

    return z

  def decode(self, tgt, encoder_output, src_mask, tgt_mask):
    # Repeat similar steps for the target sequence as done for the source sequence, using
    # the decoder embedding layer and target sequence
    z = self.decoder_embedding(tgt)
    z = self.positional_encoding(z)
    z = self.dropout(z)
    
    # Pass the decoder output along with the encoder output through each decoder layer in the decoder block
    # Each layer in the decoder applies self-attention, cross-attention with the encoder output, and then
    # as before, a feedforward network with normalizaiton and dropout applied
    z = self.decoder_block(z, encoder_output, src_mask, tgt_mask)

    return z

  def forward(self, src, tgt):
    """
    Defines the forward pass of the Transformer model. It first generates masks for the input sequences, then
    processes the sequences through the encoder and decoder stacks, and finally applies a linear layer to produce
    the output predictions.

    Parameters:
      src (Tensor): The source sequence tensor.
      tgt (Tensor): The target sequence tensor.

    Returns:
      Tensor: The output predictions of the model.
    """
    
    # Generate masks for source and target sequences
    # Source mask hides padding tokens in the source sequence
    # Target mask prevents attention to future tokens in the target sequence
    src_mask, tgt_mask = self.generate_mask(src, tgt)

    e = self.encode(src, src_mask)
    z = self.decode(tgt, e, src_mask, tgt_mask)

    # Apply a final fully connected linear layer to project the decoder output to the target vocabulary size
    # This layer converts the decoder output to logit scores for each token in the target vocabulary
    return self.linear(z)