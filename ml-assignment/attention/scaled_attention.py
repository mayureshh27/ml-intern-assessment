import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    scores = scores / np.sqrt(d_k)
    
    if mask is not None:
        if mask.ndim == 2:
            mask = np.expand_dims(mask, 0)
        
        scores = np.where(mask, -1e9, scores)
    
    attention_weights = stable_softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def stable_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    softmax_x = exp_x / (sum_exp_x + 1e-10)
    
    return softmax_x


def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask


def create_padding_mask(seq_lengths, max_len):
    batch_size = len(seq_lengths)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, length in enumerate(seq_lengths):
        if length < max_len:
            mask[i, length:] = True
    
    return mask
