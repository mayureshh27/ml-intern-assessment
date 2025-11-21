"""
Scaled Dot-Product Attention Implementation.

This module implements the core attention mechanism from "Attention Is All You Need"
using only NumPy for numerical computations.
"""

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    This is the core attention mechanism from the Transformer architecture.
    The formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        Q (np.ndarray): Query matrix of shape (batch_size, seq_len_q, d_k)
        K (np.ndarray): Key matrix of shape (batch_size, seq_len_k, d_k)
        V (np.ndarray): Value matrix of shape (batch_size, seq_len_v, d_v)
        mask (np.ndarray, optional): Mask of shape (batch_size, seq_len_q, seq_len_k)
                                     or (seq_len_q, seq_len_k).
                                     Masked positions should be True, unmasked False.
    
    Returns:
        tuple: (output, attention_weights)
            - output: Attended output of shape (batch_size, seq_len_q, d_v)
            - attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
    
    Mathematical Steps:
        1. Compute attention scores: scores = Q @ K^T
        2. Scale by sqrt(d_k): scores = scores / sqrt(d_k)
        3. Apply mask (if provided): scores[mask] = -inf
        4. Apply softmax: attention_weights = softmax(scores)
        5. Compute output: output = attention_weights @ V
    """
    # Step 1: Get the dimension of the key vectors (d_k)
    # This is used for scaling to prevent the dot products from growing too large
    d_k = Q.shape[-1]
    
    # Step 2: Compute the dot product of queries and keys
    # Q shape: (batch_size, seq_len_q, d_k)
    # K^T shape: (batch_size, d_k, seq_len_k)
    # Result shape: (batch_size, seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    
    # Step 3: Scale the scores by sqrt(d_k)
    # This prevents the dot products from becoming too large, which would
    # push the softmax into regions with extremely small gradients
    scores = scores / np.sqrt(d_k)
    
    # Step 4: Apply mask if provided
    # Masked positions are set to a very large negative number (-1e9)
    # so that after softmax, they become approximately zero
    if mask is not None:
        # Ensure mask has the right shape
        if mask.ndim == 2:
            # Broadcast mask to (batch_size, seq_len_q, seq_len_k)
            mask = np.expand_dims(mask, 0)
        
        # Set masked positions to -infinity (or a very large negative number)
        # This ensures they get zero weight after softmax
        scores = np.where(mask, -1e9, scores)
    
    # Step 5: Apply softmax to get attention weights
    # We use a numerically stable softmax implementation
    attention_weights = stable_softmax(scores, axis=-1)
    
    # Step 6: Compute the weighted sum of values
    # attention_weights shape: (batch_size, seq_len_q, seq_len_k)
    # V shape: (batch_size, seq_len_k, d_v)
    # Result shape: (batch_size, seq_len_q, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def stable_softmax(x, axis=-1):
    """
    Compute softmax in a numerically stable way.
    
    The standard softmax can overflow or underflow for large/small values.
    We subtract the maximum value before computing exp to prevent this.
    
    Args:
        x (np.ndarray): Input array
        axis (int): Axis along which to compute softmax
    
    Returns:
        np.ndarray: Softmax probabilities (same shape as input)
    """
    # Subtract the maximum value for numerical stability
    # This prevents overflow in the exp function
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    # Compute exp of shifted values
    exp_x = np.exp(x_shifted)
    
    # Normalize by the sum to get probabilities
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    
    # Avoid division by zero
    softmax_x = exp_x / (sum_exp_x + 1e-10)
    
    return softmax_x


def create_causal_mask(seq_len):
    """
    Create a causal (lower triangular) mask for autoregressive attention.
    
    This mask prevents positions from attending to future positions,
    which is essential for language modeling.
    
    Args:
        seq_len (int): Sequence length
    
    Returns:
        np.ndarray: Boolean mask of shape (seq_len, seq_len)
                   True indicates positions that should be masked
    """
    # Create a matrix where mask[i, j] = True if j > i
    # This prevents position i from attending to positions after it
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask


def create_padding_mask(seq_lengths, max_len):
    """
    Create a padding mask for variable-length sequences.
    
    Args:
        seq_lengths (np.ndarray): Array of actual sequence lengths (batch_size,)
        max_len (int): Maximum sequence length
    
    Returns:
        np.ndarray: Boolean mask of shape (batch_size, max_len)
                   True indicates padded positions
    """
    batch_size = len(seq_lengths)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, length in enumerate(seq_lengths):
        if length < max_len:
            mask[i, length:] = True
    
    return mask
