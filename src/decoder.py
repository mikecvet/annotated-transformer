from multihead import MultiHeadAttention
from position_ffnn import PositionWiseFeedForward
import torch.nn as nn

class Decoder(nn.Module):
  """
  The Decoder layer in a Transformer model is responsible for generating the output sequence based on the encoded input and 
  the partially generated output sequence up to the current step. It consists of a stack of identical layers, each containing 
  three main sub-layers: 
    - a self-attention mechanism
    - a cross-attention mechanism with the output from the Encoder as its input
    - a position-wise feedforward network
    
  Similar to the Encoder, normalization and dropout are applied for regularization.

  This implementation represents a single Decoder layer, which can be replicated to form the full Decoder stack in the 
  Transformer model architecture.
  """
  def __init__(self, dim, num_attn_heads, inner_dim, dropout_rate, device):
    """
    Initializes a Decoder layer.

    Parameters:
      dim (int): The dimensionality of input and output of this layer.
      num_attn_heads (int): The number of attention heads in both the self-attention and cross-attention mechanisms.
      inner_dim (int): The dimensionality of the hidden layer in the position-wise feedforward network.
      dropout_rate (float): The dropout rate for regularization.

    Attributes:
      self_attn (MultiHeadAttention): The multi-head self-attention mechanism, for attending to previous positions within the output sequence to predict the next symbol.
      cross_attn (MultiHeadAttention): The multi-head cross-attention mechanism, for attending to the Encoder's output.
      feed_forward (PositionWiseFeedForward): The position-wise feedforward network.
      norm1, norm2, norm3 (nn.LayerNorm): Layer normalization applied after the self-attention, cross-attention, and feedforward network, respectively.
      dropout (nn.Dropout): Dropout applied after each sub-layer for regularization.
    """
    super(Decoder, self).__init__()

    self.self_attn    = MultiHeadAttention(dim, num_attn_heads, dropout_rate, device)
    self.cross_attn   = MultiHeadAttention(dim, num_attn_heads, dropout_rate, device)
    self.feed_forward = PositionWiseFeedForward(dim, inner_dim, device)

    self.norm1   = nn.LayerNorm(dim)
    self.norm2   = nn.LayerNorm(dim)
    self.norm3   = nn.LayerNorm(dim)
    self.dropout = nn.Dropout(dropout_rate)
      
  def forward(self, dec_output, enc_output, src_mask, tgt_mask):
    """
    Defines the forward pass of the Decoder layer.

    The process involves:
    1. Self-attention with look-ahead mask: This allows each position to attend to itself and preceding positions. 
        The look-ahead mask prevents positions from attending to future positions during training. This helps 
        maintain the auto-regressive property of the Decoder.

    2. Residual connection and dropout: Similar to the Encoder, the output from the self-attention 
        mechanism is added to the original input (residual connection) and then dropout is applied for regularization.

    3. Normalization: The output is normalized using layer normalization.

    4. Cross-attention with Encoder output: The Decoder then applies cross-attention to the output of the Encoder, 
        allowing each position in the Decoder to attend to all positions in the Encoder's output.

    5. Residual connection and dropout: Again, the output from the cross-attention is added to 
        the input from the previous step (residual connection) followed by dropout.

    6. Normalization: The output is normalized.

    7. Position-wise feedforward network: The output then goes through a position-wise feedforward network.

    8. Residual connection: A final residual connection and dropout are applied.

    9. Normalization: The final normalization step produces the output of the Decoder layer.

    Parameters:
      X (Tensor):          The input tensor to the Decoder layer, typically the output from the previous Decoder layer.
      enc_output (Tensor): The output tensor from the Encoder stack.
      src_mask (Tensor):   The source mask tensor, used in the cross-attention sub-layer to mask out padding tokens 
                           from the Encoder's output.
      tgt_mask (Tensor):   The target mask tensor, used in the self-attention sub-layer to prevent the Decoder from 
                           attending to future tokens.

    Returns:
      Tensor: The output tensor of the Decoder layer.
    """
          
    dec_output = self.norm1(dec_output)

    # Self-attention with mask to prevent attention to future positions
    self_attn = self.self_attn(dec_output, dec_output, dec_output, tgt_mask)
    
    # Dropout is applied to the output of the self-attention mechanism to help prevent overfitting.
    # The output is then added to the original input `x` (residual connection) which helps mitigate vanishing gradients
    dec_output = dec_output + self.dropout(self_attn)  
    
    # Layer Normalization is performed to stabilize the network's learning and help converge faster.
    #dec_output = self.norm1(dec_output)

    dec_output = self.norm2(dec_output)
    
    # Cross-attention. The decoder's output serves as the query, and the encoder output serves as both key and value.
    # This allows the decoder to focus on relevant parts of the input sequence.
    cross_attn = self.cross_attn(dec_output, enc_output, enc_output, src_mask)
    
    dec_output = dec_output + self.dropout(cross_attn)
    
    # After adding the cross-attention output back to the original input (residual connection), apply second layer normalization.
    #dec_output = self.norm2(dec_output)

    dec_output = self.norm3(dec_output)
    
    # The output from the previous normalization step is passed through the position-wise FFNN
    y = self.feed_forward(dec_output)
    
    dec_output = dec_output + self.dropout(y)

    return dec_output
    
    # The final step in the decoder layer is to normalize the output after the last residual connection.
    #return self.norm3(dec_output)  # Apply the final layer normalization