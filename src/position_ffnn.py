import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
  """
  Implements the position-wise feedforward network (FFN) layer for a Transformer model. Each layer in the Transformer architecture
  includes a multi-head attention mechanism followed by a position-wise FFN. The FFN is applied to each position separately and
  identically. This means the same operations are performed on every position in the sequence, but independently, allowing the model
  to consider the context provided by the attention mechanism and further process the information through a non-linear transformation.

  The FFN consists of two linear transformations with a ReLU (or similar non-linear) activation in between. This design allows the
  model to capture complex relationships within the data by introducing non-linearity into the processing pipeline, enhancing the
  model's representational power.

  https://arxiv.org/pdf/1706.03762.pdf # 3.3
  """
  def __init__(self, dimensionality, inner_dimensionality, device):
    """
    Initializes the PositionWiseFeedForward module.

    Parameters:
      dimensionality (int):       The dimensionality of the input and output of the FFN. Should match the model's size.
      inner_dimensionality (int): The dimensionality of the hidden layer within the FFN. This is usually larger than `dimension`,
                              allowing the network to expand the representation before applying the second linear transformation.

    Attributes:
      linear1 (nn.Linear): The first linear transformation layer, mapping from `dimension` to `inner_dimension`.
      gelu      (nn.GELU): The GELU activation function applied after the first linear transformation.
      linear2 (nn.Linear): The second linear transformation layer, mapping back from `inner_dimension` to `dimension`.
    """
    super(PositionWiseFeedForward, self).__init__()

    self.linear1 = nn.Linear(dimensionality, inner_dimensionality).to(device)
    self.linear2 = nn.Linear(inner_dimensionality, dimensionality).to(device)
    self.gelu = nn.GELU()

  def forward(self, X):
    """
    Applies the position-wise FFN to the input tensor `X`.

    Parameters:
      X (Tensor): The input tensor to the FFN. Expected shape `[batch_size, seq_length, dimensionality]`.

    Returns:
      Tensor: The output tensor after applying the FFN, with the same shape as the input tensor.
    """
    X = self.linear1(X)
    X = self.gelu(X)
    
    return self.linear2(X)