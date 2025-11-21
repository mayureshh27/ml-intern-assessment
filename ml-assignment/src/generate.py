from .ngram_model import TrigramModel
import os
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    alice_path = os.path.join(data_dir, 'alice_in_wonderland.txt')
    example_path = os.path.join(data_dir, 'example_corpus.txt')
    
    if os.path.exists(alice_path):
        corpus_path = alice_path
        corpus_name = "Alice's Adventures in Wonderland"
    elif os.path.exists(example_path):
        corpus_path = example_path
        corpus_name = "Example Corpus"
        print("⚠ Using example corpus. For better results, run: python download_corpus.py")
        print()
    else:
        print("❌ Error: No corpus found!")
        print("Please run: python download_corpus.py")
        sys.exit(1)
    
    print(f"📚 Loading corpus: {corpus_name}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Corpus size: {len(text)} characters, {len(text.split())} words")
    print()
    
    print("🔧 Training trigram model...")
    model = TrigramModel()
    model.fit(text)
    
    print(f"   Vocabulary size: {len(model.vocabulary)} unique words")
    print(f"   Trigram count: {sum(len(v2) for v1 in model.trigram_counts.values() for v2 in v1.values())}")
    print()
    
    print("✨ Generating text samples:")
    print("=" * 70)
    
    num_samples = 5
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        print("-" * 70)
        generated_text = model.generate(max_length=50)
        print(generated_text)
    
    print("\n" + "=" * 70)
    print("\n✓ Generation complete!")
    print("\nTip: Run this script multiple times to see different generated texts.")


if __name__ == "__main__":
    main()
