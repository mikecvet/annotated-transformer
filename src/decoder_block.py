from decoder import Decoder
import torch.nn as nn

class DecoderBlock(nn.Module):
  """
  A DecoderBlock is a sequence of Transformer decoder layers encapsulated as a single module.
  This class manages a stack of multiple decoder layers, where each layer individually
  performs a series of operations involving self-attention, cross-attention with encoder outputs,
  and feed-forward neural network transformations.

  The DecoderBlock facilitates the sequential processing of these layers, where the output of
  one layer becomes the input to the next, effectively building a deep stack of decoder layers.

  https://arxiv.org/pdf/1706.03762.pdf # 3.1

  Parameters:
  - dim (int):             Dimensionality of the input and output tokens/sequences.
  - num_attn_heads (int):  Number of attention heads in each multi-head attention mechanism.
  - num_layers (int):      Number of Decoder layers to stack.
  - inner_dim (int):       Dimensionality of the inner feed-forward neural network.
  - dropout_rate (float):  Dropout rate to use within each Decoder layer for regularization.
  - device (torch.device): The device (CPU or GPU) on which the tensors should be allocated.

  Attributes:
  - decoder_layers (nn.ModuleList): A list containing the stacked Decoder layers.
  """
  def __init__(self, dim, num_attn_heads, num_layers, inner_dim, dropout_rate, device):
    super(DecoderBlock, self).__init__()
    self.decoder_layers = nn.ModuleList(
      [Decoder(dim, num_attn_heads, inner_dim, dropout_rate, device) for _ in range(num_layers)]
    )

  def forward(self, X, enc_output, src_mask, tgt_mask):
    """
    Defines the forward pass of the DecoderBlock through the sequence of decoder layers.
    Each layer processes the input sequentially, using both self-attention on the input 
    and cross-attention on the encoder's output.

    Args:
    - X (Tensor):           The input tensor to the decoder block with shape [seq_len, batch_size, dim].
    - enc_output (Tensor):  The output tensor from the encoder block used in cross-attention,
        with shape [src_seq_len, batch_size, dim].
    - src_mask (Tensor):    The source mask tensor used to mask out invalid source positions
        during cross-attention, with shape compatible with the encoder output dimensions.
    - tgt_mask (Tensor):    The target mask tensor used to prevent future peeking in self-attention
        and ensure that predictions for a sequence position can depend only on earlier positions,
        typically a triangular matrix.

    Returns:
    - Tensor: The output tensor after processing through all decoder layers, maintaining the
      input shape [seq_len, batch_size, dim].
    """
    for decoder in self.decoder_layers:
      X = decoder(X, enc_output, src_mask, tgt_mask)
    
    return X
