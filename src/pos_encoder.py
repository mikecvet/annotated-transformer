import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  """
  Implements positional encoding to provide information about the order of sequence elements to the Transformer model. 
  Transformers process elements of a sequence in parallel, thus lack an inherent understanding of their ordering.
  Positional encodings are added to the input embeddings at the bottoms of the model architecture to inject information 
  about the position of the sequence elements. This class generates a matrix of sinusoidal positional encodings and adds 
  them to input embeddings.

  https://arxiv.org/pdf/1706.03762.pdf # 3.5
  """
  def __init__(self, dim, max_seq_len, device):
    """
    Initializes the PositionalEncoder module.

    Parameters:
      dim (int): The dimensionality of the input embeddings. The Transformer model's size.
      max_seq_len (int): The maximum length of the input sequences.
      device: (torch.Device): The device to be used for computation

    Attributes:
      positional_encodings (Tensor): A precomputed tensor with positional encodings for each position up to `max_seq_len` 
      and `dim` dimensions.
    """
    super(PositionalEncoding, self).__init__()
    
    # The positional encodings matrix
    self.encodings = torch.zeros(max_seq_len, dim, device=device)
    # Position indices for the sequence
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    # Scale for position indices
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    
    # If i is even: PE(pos, i) = sin(pos / (10000 * (2i / dimension)))
    self.encodings[:, 0::2] = torch.sin(position * div_term)
    # If i is odd: PE(pos, i) = cos(pos / (10000 * (2i / dimension)))
    self.encodings[:, 1::2] = torch.cos(position * div_term)
    
    # Registers `encodings` as a buffer, part of the module's state
    self.register_buffer('positional_encodings', self.encodings.unsqueeze(0))
      
  def forward(self, X):
    """
    Adds the positional encodings to the input embeddings.

    Parameters:
      X (Tensor): The input embeddings tensor.

    Returns:
      Tensor: The input embeddings tensor with added positional encodings.
    """
    seq_len = X.size(1)

    # Ensure that self.encodings are correctly aligned with x in dimensions
    # Match seq_len with x and ensure embedding_dim aligns
    if self.encodings.shape[0] < seq_len:
        # Error handling or dynamic resizing of self.encodings might be necessary here
        raise ValueError("Positional encodings length is shorter than input sequence length.")
    
    # Correct slicing to ensure dimensions match: [1, seq_len, embedding_dim]
    # This slicing notation collects `seq_len` rows and all column from the encodings tensor, then
    # inserts a single empty dimension at position 0.
    pos_encodings = self.encodings[:seq_len, :].unsqueeze(0)
    
    # Expand positional encodings to match the batch size of X
    # Since broadcasting handles the batch size, explicit expansion might not be necessary
    return X + pos_encodings