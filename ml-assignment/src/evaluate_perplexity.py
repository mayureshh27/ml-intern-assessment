from .ngram_model import TrigramModel
import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    alice_path = os.path.join(data_dir, 'alice_in_wonderland.txt')
    
    if not os.path.exists(alice_path):
        print("❌ Error: alice_in_wonderland.txt not found!")
        print("Please run: python -m src.download_corpus")
        return
    
    print("=" * 70)
    print("  PERPLEXITY EVALUATION DEMONSTRATION")
    print("=" * 70)
    
    with open(alice_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    split_point = int(len(full_text) * 0.8)
    train_text = full_text[:split_point]
    test_text = full_text[split_point:]
    
    print(f"\n📚 Corpus: Alice's Adventures in Wonderland")
    print(f"   Total size: {len(full_text):,} characters")
    print(f"   Training set: {len(train_text):,} characters (80%)")
    print(f"   Test set: {len(test_text):,} characters (20%)")
    
    print("\n🔧 Training trigram model...")
    model = TrigramModel()
    model.fit(train_text)
    
    print(f"   Vocabulary size: {len(model.vocabulary):,} unique words")
    print(f"   Trigram count: {sum(len(v2) for v1 in model.trigram_counts.values() for v2 in v1.values()):,}")
    
    print("\n📊 Calculating perplexity...")
    
    train_perplexity = model.calculate_perplexity(train_text[:10000])
    print(f"\n   Training set perplexity: {train_perplexity:.2f}")
    print(f"   (Lower is better - measures how well model fits training data)")
    
    test_perplexity = model.calculate_perplexity(test_text[:10000])
    print(f"\n   Test set perplexity: {test_perplexity:.2f}")
    print(f"   (Measures generalization to unseen text)")
    
    print("\n📈 Interpretation:")
    if test_perplexity < train_perplexity * 1.5:
        print("   ✅ Good generalization - test perplexity is close to training")
    elif test_perplexity < train_perplexity * 2.5:
        print("   ⚠️  Moderate generalization - some overfitting may be present")
    else:
        print("   ❌ Poor generalization - significant overfitting detected")
    
    print(f"\n   Perplexity ratio (test/train): {test_perplexity/train_perplexity:.2f}x")
    
    print("\n" + "=" * 70)
    print("  Perplexity Evaluation Complete!")
    print("=" * 70)
    
    print("\n💡 What is Perplexity?")
    print("   - Measures how 'surprised' the model is by new text")
    print("   - Lower perplexity = better predictions")
    print("   - Perplexity of N means model is as uncertain as choosing")
    print("     randomly from N equally likely words")
    print(f"   - This model's test perplexity of {test_perplexity:.0f} means it's as")
    print(f"     uncertain as choosing from ~{test_perplexity:.0f} words on average")


if __name__ == "__main__":
    main()
