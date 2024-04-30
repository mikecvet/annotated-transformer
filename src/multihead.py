import numpy as np
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  """
  Implements a Multi-Head Attention mechanism. This allows the model to jointly attend to information from different
  representation subspaces at different positions. With multiple attention heads, the model can capture a large 
  diversity of relationships within the data.

  The multi-head attention mechanism performs attention operations in parallel, each with its own set of linear transformations to
  produce queries, keys, and values. The outputs of these parallel operations are then combined and passed through a final linear
  layer to produce the result.

  https://arxiv.org/pdf/1706.03762.pdf # 3.2
  """  

  def __init__(self, dim, num_attn_heads, dropout_rate, device):
    """
    Initializes the MultiHeadAttention module.

    Parameters:
      dim (int): The dimensionality of the input and output of this layer.
      num_attn_heads (int): The number of attention heads.

    Raises:
      AssertionError: If `dimensionality` is not divisible by `num_attn_heads`.
    """
    super(MultiHeadAttention, self).__init__()

    # Ensure that the model dimension (d_model) is divisible by the number of heads
    assert dim % num_attn_heads == 0, f"dimensionality {dim} must be divisible by num heads {num_attn_heads}"
    
    self.dim = dim # Model's dimension
    self.num_attn_heads = num_attn_heads # Number of attention heads
    self.kqv_dim = dim // num_attn_heads # Dimension of each head's key, query, and value
    
    self.W_q = nn.Linear(dim, dim, bias=False).to(device) # Query linear transformation
    self.W_k = nn.Linear(dim, dim, bias=False).to(device) # Key linear transformation
    self.W_v = nn.Linear(dim, dim, bias=False).to(device) # Value linear transformation
    self.W_o = nn.Linear(dim, dim).to(device) # Output linear transformation

    self.dropout = nn.Dropout(dropout_rate)

    nn.init.normal_(self.W_q.weight, std=np.sqrt(2 / (self.dim * 2 // self.num_attn_heads)))
    nn.init.normal_(self.W_k.weight, std=np.sqrt(2 / (self.dim * 2 // self.num_attn_heads)))
    nn.init.normal_(self.W_v.weight)
    nn.init.zeros_(self.W_o.bias)
  
  def scaled_dot_product_attention(self, Q, K, V, mask=None):
    """
    Computes the scaled dot-product attention over the supplied queries, keys, and values.

    https://arxiv.org/pdf/1706.03762.pdf # 3.2.1

    Parameters:
      Q (Tensor): Queries tensor.
      K (Tensor): Keys tensor.
      V (Tensor): Values tensor.
      mask (Optional[Tensor]): An optional mask to exclude certain positions from attending to others.

    Returns:
      Tensor: The result of the attention operation.
    """

    # Dot product attention scores are obtained by computing the dot product between a query (Q) and all keys (K). 
    # This measures the compatibility similarity between each query and each key. The higher the dot product score, the more r
    # elevant the key is to the query.
    #
    # In order to accomplish this, there must be a transposition of the keys tensor for matrix compatibility.
    # Here, we change K's shape from [batch_size, num_heads, seq_length, depth] to [batch_size, num_heads, depth, seq_length] 
    # Now multiplication with Q ([batch_size, num_heads, seq_length, depth]) is aligned correctly: 
    #  the depth of Q is multiplied by the depth of K (now the second last dimension due to the transpose), resulting 
    #  in a tensor of shape [batch_size, num_heads, seq_length, seq_length]
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.kqv_dim)
    
    # Apply mask if provided (useful for preventing attention to certain parts like padding)
    if mask is not None:
      attn_scores = attn_scores.masked_fill(mask == 0, 1e-10)
    
    # Calculate softmax to compute probabilities across attention scores
    attn_probs = torch.softmax(attn_scores, dim=-1)

    attn_probs = self.dropout(attn_probs)
    
    # Apply attention probabilities against the values tensor
    return attn_probs @ V # double check

  def split_attn_heads(self, X):
    """
    Splits the input tensor into multiple heads for parallel processing.

    Parameters:
      X (Tensor): The input tensor.

    Returns:
      Tensor: The reshaped tensor for multi-head processing.
    """

    # Reshape the input to have num_heads for multi-head attention, from 
    # (batch size, sequence length, num heads, dimension) to (batch size, num heads, sequence length, dimension)
    batch_size, seq_length, _ = X.size()
    return X.view(
      batch_size, seq_length, self.num_attn_heads, self.kqv_dim
    ).transpose(1, 2) # Swap seq_length and num_attn_heads dimensions
      
  def combine_heads(self, X):
    """
    Inverse of split_attn_heads(); combines results of attention processing back into a single tensor of shape
    (batch_size, seq_length, dimension)

    Parameters:
      X (Tensor): The multi-head tensor.

    Returns:
      Tensor: The combined tensor.
    """

    batch_size, _, seq_length, _ = X.size()

    # Swap num_attn_heads and seq_length dimensions so that the result is once again
    # in the shape of (batch size, sequence length, dimension)
    return X.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dim)
      
  def forward(self, Q, K, V, mask=None):
    """
    Attention computation. Applies linear layers to the query, key and values inputs, then splits 
    into multiple heads. Scaled dot-product attention is computed on the results, whise heads are then reconstituted
    and finally passed through the final linear output layer.

    https://arxiv.org/pdf/1706.03762.pdf # 3.2.2

    Parameters:
      Q (Tensor): Query tensor.
      K (Tensor): Key tensor.
      V (Tensor): Value tensor.
      mask (Optional[Tensor]): An optional mask tensor.

    Returns:
      Tensor: The output of the multi-head attention operation.
    """

    Q = self.split_attn_heads(self.W_q(Q))
    K = self.split_attn_heads(self.W_k(K))
    V = self.split_attn_heads(self.W_v(V))
    
    attn = self.scaled_dot_product_attention(Q, K, V, mask)
    combined = self.combine_heads(attn)
    
    return self.W_o(combined)