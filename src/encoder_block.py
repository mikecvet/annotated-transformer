import torch.nn as nn
from encoder import Encoder

class EncoderBlock(nn.Module):
  """
  An EncoderBlock encapsulates a stack of Encoder layers in a Transformer model,
  each performing multi-head self-attention and feed-forward operations. This block
  manages the sequence of encoder layers, processing the input sequentially through
  each layer in the stack.

  The EncoderBlock serves to transform the input sequences into a continuous representation
  that captures the relationships between different elements of the sequence, based on
  the self-attention mechanism. This representation is then used by the Transformer's decoder
  or further processed depending on the specific application.

  https://arxiv.org/pdf/1706.03762.pdf # 3.1

  Parameters:
  - dim (int):             Dimensionality of the input and output tokens/sequences.
  - num_attn_heads (int):  Number of attention heads in each multi-head attention mechanism.
  - num_layers (int):      Number of Encoder layers to stack.
  - inner_dim (int):       Dimensionality of the inner feed-forward neural network.
  - dropout_rate (float):  Dropout rate to use within each Encoder layer for regularization.
  - device (torch.device): The device (CPU or GPU) on which the tensors should be allocated.

  Attributes:
  - encoder_layers (nn.ModuleList): A list of Encoder layers that are applied sequentially.
  """
  def __init__(self, dim, num_attn_heads, num_layers, inner_dim, dropout_rate, device):
    super(EncoderBlock, self).__init__()
    self.encoder_layers = nn.ModuleList(
      [Encoder(dim, num_attn_heads, inner_dim, dropout_rate, device) for _ in range(num_layers)]
    )

  def forward(self, X, mask):
    """
    Defines the computation performed at every call of the EncoderBlock.
    It sequentially processes the input through each of the Encoder layers using self-attention
    and feed-forward networks.

    Args:
    - X (Tensor):    The input tensor to the encoder block with shape [seq_len, batch_size, dim].
    - mask (Tensor): The mask tensor used in self-attention to prevent attention to certain positions,
      typically used for padding. The shape should be compatible with the dimensions of X.

    Returns:
    - Tensor: The output tensor after processing through all encoder layers, maintaining the
      input shape [seq_len, batch_size, dim].
    """
    for encoder in self.encoder_layers:
      X = encoder(X, mask)
    
    return X
