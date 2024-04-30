from multihead import MultiHeadAttention
from position_ffnn import PositionWiseFeedForward
import torch.nn as nn

class Encoder(nn.Module):
  """
  The Encoder layer is responsible for processing the input sequence within a Transformer architecture.
  It consists of a stack of identical layers, each containing two main sub-layers: 
    - a multi-head self-attention mechanism
    - a position-wise feedforward neural network 
    
  Normalization and dropout are applied for regularization and to prevent overfitting.

  This implementation represents a single Encoder layer, which can be replicated to form the full Encoder stack as per the 
  Transformer model's requirements.
  """

  def __init__(self, dim, num_attn_heads, inner_dim, dropout_rate, device):
    """
    Initializes an Encoder layer.

    Parameters:
      dim (int): The dimensionality of input and output of this layer.
      num_attn_heads (int): The number of attention heads in the multi-head attention mechanism.
      inner_dim (int): The dimensionality of the hidden layer in the position-wise feedforward network.
      dropout_rate (float): The dropout rate to use for regularization.

    Attributes:
      self_attn (MultiHeadAttention): The multi-head self-attention mechanism.
      feed_forward (PositionWiseFeedForward): The position-wise feedforward network.
      norm1 (nn.LayerNorm): The first layer normalization applied after the self-attention and before the residual connection.
      norm2 (nn.LayerNorm): The second layer normalization applied after the feedforward network and before the second residual connection.
      dropout (nn.Dropout): The dropout layer for after self-attention and feedforward networks for regularization.
    """
    super(Encoder, self).__init__()
    self.self_attn = MultiHeadAttention(dim, num_attn_heads, dropout_rate, device)
    self.pwffnn = PositionWiseFeedForward(dim, inner_dim, device)

    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)
    self.dropout = nn.Dropout(dropout_rate)
    
  def forward(self, x, mask):
    """
    Defines the forward pass of the Encoder layer with an input tensor `x` and a mask `mask`.

    The process within the Encoder layer involves several key steps:
    1. **Self-attention**: The input tensor `x` is processed through a multi-head self-attention mechanism. 
        This allows each position in the input to attend to all positions in the same input sequence. 
        The self-attention mechanism takes the input tensor `x` as queries, keys, and values, along with an optional mask to prevent 
        attention to certain positions (useful for ignoring padding in sequences).

    2. **Residual connection and dropout after self-attention**: The output of the self-attention mechanism is then added back to the 
        original input tensor `x` to form a residual connection. This helps in mitigating the vanishing gradient problem by allowing 
        gradients to flow directly through the network. After the residual connection, dropout is applied as a form of regularization 
        to prevent overfitting by randomly zeroing out a fraction of the outputs.

    3. **Normalization after self-attention**: The result from the previous step is then normalized using layer normalization. Layer 
        normalization is applied across all features (the dimension of the input tensor) for each data point in the batch independently. 
        This normalization step helps in stabilizing the learning process and accelerates the training of deep neural networks.

    4. **Position-wise feedforward network**: The normalized output is then passed through a position-wise feedforward network (FFN). 
        The FFN applies two linear transformations with a ReLU activation in between. Unlike the self-attention mechanism, the FFN is 
        applied independently to each position.

    5. **Residual connection and dropout after the feedforward network**: Similar to step 2, the output of the FFN is added back to its 
        input (before the FFN), forming another residual connection. Dropout is then applied again for regularization.

    6. **Normalization after the feedforward network**: Finally, the output from the previous step is normalized using another layer 
        normalization step. This produces the final output tensor of the Encoder layer.

    Parameters:
        x (Tensor): The input tensor to the Encoder layer. Shape: [batch size, sequence length, dimension].
        mask (Tensor): The mask tensor for the self-attention mechanism. Shape: [batch size, 1, sequence length].

    Returns:
        Tensor: The output tensor of the Encoder layer. Shape: [batch size, sequence length, dimension].
    """

    # Normalization before self-attention
    x = self.norm1(x)

    # Self-attention
    attn = self.self_attn(x, x, x, mask)
    
    # Residual connection + dropout after self-attention
    x = x + self.dropout(attn)
    
    # Normalization after self-attention
    #x = self.norm1(x)

    x = self.norm2(x)
    
    # Position-wise feedforward network
    y = self.pwffnn(x)
    
    # Residual connection + dropout after FFNN
    x = x + self.dropout(y)

    return x

    # Final normalization after FFNN
    #return self.norm2(x)