import pytest
import numpy as np
from attention.scaled_attention import (
    scaled_dot_product_attention,
    stable_softmax,
    create_causal_mask,
    create_padding_mask
)


def test_basic_attention_shapes():
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 8
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_v)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)


def test_attention_weights_sum_to_one():
    batch_size = 2
    seq_len = 5
    d_k = 4
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    _, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    weight_sums = attention_weights.sum(axis=-1)
    
    np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-5)


def test_attention_weights_non_negative():
    batch_size = 1
    seq_len = 3
    d_k = 4
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    _, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    assert np.all(attention_weights >= 0)
    assert np.all(attention_weights <= 1)


def test_causal_mask_structure():
    seq_len = 5
    mask = create_causal_mask(seq_len)
    
    assert mask.shape == (seq_len, seq_len)
    
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[i, j] == True
            else:
                assert mask[i, j] == False


def test_attention_with_causal_mask():
    batch_size = 1
    seq_len = 4
    d_k = 4
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    mask = create_causal_mask(seq_len)
    _, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert attention_weights[0, i, j] < 1e-6


def test_padding_mask_creation():
    seq_lengths = np.array([3, 5, 2])
    max_len = 5
    
    mask = create_padding_mask(seq_lengths, max_len)
    
    assert mask.shape == (3, 5)
    
    assert np.all(mask[0, :3] == False)
    assert np.all(mask[0, 3:] == True)
    
    assert np.all(mask[1, :] == False)
    
    assert np.all(mask[2, :2] == False)
    assert np.all(mask[2, 2:] == True)


def test_stable_softmax_numerical_stability():
    x = np.array([[1000, 1001, 1002]])
    
    result = stable_softmax(x, axis=-1)
    
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-5)


def test_stable_softmax_output_range():
    x = np.random.randn(2, 5)
    
    result = stable_softmax(x, axis=-1)
    
    assert np.all(result >= 0)
    assert np.all(result <= 1)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-5)


def test_attention_deterministic():
    np.random.seed(42)
    batch_size = 1
    seq_len = 3
    d_k = 4
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    output1, weights1 = scaled_dot_product_attention(Q, K, V)
    
    np.random.seed(42)
    Q2 = np.random.randn(batch_size, seq_len, d_k)
    K2 = np.random.randn(batch_size, seq_len, d_k)
    V2 = np.random.randn(batch_size, seq_len, d_k)
    
    output2, weights2 = scaled_dot_product_attention(Q2, K2, V2)
    
    np.testing.assert_array_equal(output1, output2)
    np.testing.assert_array_equal(weights1, weights2)


def test_attention_with_different_seq_lengths():
    batch_size = 1
    seq_len_q = 3
    seq_len_k = 5
    d_k = 4
    d_v = 6
    
    Q = np.random.randn(batch_size, seq_len_q, d_k)
    K = np.random.randn(batch_size, seq_len_k, d_k)
    V = np.random.randn(batch_size, seq_len_k, d_v)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len_q, d_v)
    assert attention_weights.shape == (batch_size, seq_len_q, seq_len_k)


def test_attention_scaling():
    batch_size = 1
    seq_len = 3
    d_k = 64
    
    Q = np.ones((batch_size, seq_len, d_k))
    K = np.ones((batch_size, seq_len, d_k))
    V = np.random.randn(batch_size, seq_len, d_k)
    
    _, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    np.testing.assert_allclose(attention_weights.sum(axis=-1), 1.0, rtol=1e-5)
    
    expected_uniform = 1.0 / seq_len
    np.testing.assert_allclose(attention_weights[0], expected_uniform, rtol=0.1)


def test_identity_attention():
    batch_size = 1
    seq_len = 3
    d_k = 4
    
    Q = np.eye(d_k).reshape(1, d_k, d_k)[:, :seq_len, :]
    K = np.eye(d_k).reshape(1, d_k, d_k)[:, :seq_len, :]
    V = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_k)
    np.testing.assert_allclose(attention_weights.sum(axis=-1), 1.0, rtol=1e-5)


def test_zero_query():
    batch_size = 1
    seq_len = 3
    d_k = 4
    
    Q = np.zeros((batch_size, seq_len, d_k))
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_k)
    np.testing.assert_allclose(attention_weights.sum(axis=-1), 1.0, rtol=1e-5)
    
    expected_uniform = 1.0 / seq_len
    np.testing.assert_allclose(attention_weights[0], expected_uniform, rtol=0.01)
