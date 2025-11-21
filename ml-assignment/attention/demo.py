import numpy as np
from .scaled_attention import (
    scaled_dot_product_attention,
    create_causal_mask,
    create_padding_mask
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_matrix(name, matrix, precision=4):
    print(f"\n{name}:")
    if matrix.ndim == 3:
        for i, mat in enumerate(matrix):
            print(f"  Batch {i}:")
            print_2d_matrix(mat, precision, indent="    ")
    else:
        print_2d_matrix(matrix, precision, indent="  ")


def print_2d_matrix(matrix, precision=4, indent=""):
    for row in matrix:
        formatted_row = [f"{val:>{precision+3}.{precision}f}" for val in row]
        print(indent + " ".join(formatted_row))


def demo_basic_attention():
    print_section("Demo 1: Basic Scaled Dot-Product Attention")
    
    np.random.seed(42)
    
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Key/Query dimension (d_k): {d_k}")
    print(f"  Value dimension (d_v): {d_v}")
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    print(f"\nInput shapes:")
    print(f"  Q (Query): {Q.shape}")
    print(f"  K (Key):   {K.shape}")
    print(f"  V (Value): {V.shape}")
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput shapes:")
    print(f"  Output:            {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    
    print_matrix("Attention Weights (Batch 0)", attention_weights[0])
    
    weight_sums = attention_weights.sum(axis=-1)
    print(f"\nAttention weight sums (should all be ~1.0):")
    print(f"  {weight_sums[0]}")
    
    print_matrix("Output (Batch 0)", output[0])
    
    return output, attention_weights


def demo_causal_attention():
    print_section("Demo 2: Causal Attention (with Masking)")
    
    np.random.seed(42)
    
    batch_size = 1
    seq_len = 5
    d_k = 4
    d_v = 4
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Dimensions: d_k={d_k}, d_v={d_v}")
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    mask = create_causal_mask(seq_len)
    
    print("\nCausal Mask (True = masked, False = visible):")
    print("  (Each position can only attend to itself and previous positions)")
    print_2d_matrix(mask.astype(int), precision=0, indent="  ")
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print_matrix("Attention Weights (with causal mask)", attention_weights[0])
    
    print("\nNotice:")
    print("  - Each row sums to 1.0")
    print("  - Upper triangle is all zeros (can't attend to future)")
    print("  - Position i can only attend to positions 0 through i")
    
    return output, attention_weights


def demo_attention_interpretation():
    print_section("Demo 3: Interpretable Attention Example")
    
    print("\nScenario: Simple sequence with clear patterns")
    print("  We'll create Q, K, V such that certain positions attend to others")
    
    batch_size = 1
    seq_len = 3
    d_k = 2
    d_v = 2
    
    Q = np.array([[[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]]])
    
    K = np.array([[[0.0, 1.0],
                   [1.0, 0.0],
                   [1.0, 1.0]]])
    
    V = np.array([[[1.0, 0.0],
                   [0.0, 1.0],
                   [0.5, 0.5]]])
    
    print_matrix("Query (Q)", Q[0], precision=1)
    print_matrix("Key (K)", K[0], precision=1)
    print_matrix("Value (V)", V[0], precision=1)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print_matrix("Attention Weights", attention_weights[0])
    print_matrix("Output", output[0])
    
    print("\nInterpretation:")
    print("  - Position 0's query [1,0] matches best with position 1's key [1,0]")
    print("  - So position 0 attends most to position 1")
    print("  - The output is a weighted combination of all values")


def main():
    print("\n" + "=" * 70)
    print("  SCALED DOT-PRODUCT ATTENTION DEMONSTRATION")
    print("  Implementation using only NumPy")
    print("=" * 70)
    
    demo_basic_attention()
    demo_causal_attention()
    demo_attention_interpretation()
    
    print("\n" + "=" * 70)
    print("  All demonstrations complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Attention weights always sum to 1.0 (probability distribution)")
    print("  2. Masking sets certain positions to zero weight")
    print("  3. Output is a weighted combination of value vectors")
    print("  4. Scaling by sqrt(d_k) prevents gradient issues")
    print("\n")


if __name__ == "__main__":
    main()
