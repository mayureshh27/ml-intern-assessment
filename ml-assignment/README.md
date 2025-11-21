# Trigram Language Model

This directory contains a complete implementation of a trigram (N=3) language model built from scratch in Python.

## 📋 Overview

The trigram model learns word sequences from text and generates new text that mimics the style and structure of the training corpus. This implementation uses:
- **Nested dictionaries** for efficient n-gram storage
- **Laplace smoothing** for handling unseen contexts
- **Probabilistic sampling** for diverse text generation
- **Sentence-level padding** to respect natural language structure

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Navigate to the project root
cd ml-intern-assessment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
./venv/Scripts/activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Training Corpus

```bash
cd ml-assignment/src
python download_corpus.py
```

This downloads "Alice's Adventures in Wonderland" from Project Gutenberg (~145KB, ~26K words).

### 3. Generate Text

```bash
python generate.py
```

This will:
1. Load the corpus
2. Train the trigram model
3. Generate 5 sample texts

**Expected Output:**
```
📚 Loading corpus: Alice's Adventures in Wonderland
   Corpus size: 144599 characters, 26525 words

🔧 Training trigram model...
   Vocabulary size: 2583 unique words
   Trigram count: 23603

✨ Generating text samples:
======================================================================

Sample 1:
----------------------------------------------------------------------
and certainly there was generally a frog and both the hedgehogs...
```

## 📁 Project Structure

```
ml-assignment/
├── src/
│   ├── ngram_model.py      # Core trigram model implementation
│   ├── utils.py            # Text preprocessing utilities
│   ├── generate.py         # Text generation script
│   └── download_corpus.py  # Corpus download script
├── tests/
│   └── test_ngram.py       # Comprehensive test suite
├── attention/              # Optional: Scaled dot-product attention
│   ├── scaled_attention.py # Attention implementation
│   └── demo.py             # Attention demonstration
├── data/
│   └── alice_in_wonderland.txt  # Training corpus (after download)
├── evaluation.md           # Design decisions and analysis
└── README.md               # This file
```

## 🧪 Running Tests

```bash
# From ml-assignment directory
cd ..
python -m pytest ml-assignment/tests/test_ngram.py -v
```

**Test Coverage:**
- Basic fit and generate functionality
- Empty and short text handling
- Trigram counting accuracy
- Probability normalization
- Vocabulary building
- Sentence boundary handling
- Deterministic generation with seeds
- Max length constraints
- Error handling

## 🎯 Usage Examples

### Basic Usage

```python
from src.ngram_model import TrigramModel

# Create and train model
model = TrigramModel()
with open('data/alice_in_wonderland.txt', 'r') as f:
    text = f.read()
model.fit(text)

# Generate text
generated = model.generate(max_length=50)
print(generated)
```

### With Custom Parameters

```python
# Use different smoothing parameter
model = TrigramModel(smoothing_alpha=0.1)
model.fit(text)

# Generate with seed for reproducibility
text1 = model.generate(max_length=30, seed=42)
text2 = model.generate(max_length=30, seed=42)
assert text1 == text2  # Same output
```

### Calculate Probabilities

```python
# Get probability of a specific trigram
prob = model.get_trigram_probability('<START>', '<START>', 'alice')
print(f"P(alice | <START> <START>) = {prob:.4f}")
```

## 🔬 Optional: Scaled Dot-Product Attention

This implementation also includes a from-scratch implementation of the attention mechanism from "Attention Is All You Need".

```bash
cd ml-assignment/attention
python demo.py
```

This demonstrates:
1. Basic attention computation
2. Causal (masked) attention
3. Interpretable attention examples

## 📊 Model Performance

### Training Statistics
- **Corpus**: Alice's Adventures in Wonderland
- **Training time**: <1 second
- **Vocabulary size**: 2,583 unique words
- **Unique trigrams**: 23,603
- **Memory usage**: ~2-3 MB

### Generation Quality
- ✅ Grammatically plausible sentences
- ✅ Vocabulary consistent with source
- ✅ Proper sentence boundaries
- ✅ Diverse outputs (probabilistic sampling)
- ⚠️ Limited long-range coherence (expected for trigrams)

## 🛠️ Design Choices

See [evaluation.md](evaluation.md) for detailed explanations of:
- N-gram storage structure
- Text cleaning strategy
- Padding approach
- Smoothing technique
- Probability sampling method
- Complexity analysis

## 📝 Key Implementation Details

### Text Preprocessing
1. Convert to lowercase
2. Preserve sentence-ending punctuation (. ! ?)
3. Remove other punctuation
4. Split into sentences
5. Add `<START>` and `<END>` tokens

### Probability Calculation
```
P(w3|w1,w2) = (count(w1,w2,w3) + α) / (count(w1,w2) + α * |V|)
```
Where α = 0.01 (Laplace smoothing parameter)

### Generation Algorithm
1. Start with context: `[<START>, <START>]`
2. For each position:
   - Get possible next words for current context
   - Convert counts to probabilities (with smoothing)
   - Sample next word from probability distribution
   - Update context (shift window)
3. Stop at `<END>` token or max_length

## 🐛 Troubleshooting

### Corpus Download Fails
If `download_corpus.py` fails due to network issues:
1. Manually download from: https://www.gutenberg.org/ebooks/11
2. Save as `data/alice_in_wonderland.txt`
3. Remove Project Gutenberg header/footer

### Import Errors
Make sure you're running from the correct directory and the virtual environment is activated.

### Tests Fail
Ensure all dependencies are installed:
```bash
pip install -r ../requirements.txt
```

## 📚 References

- **N-gram Language Models**: Jurafsky & Martin, "Speech and Language Processing"
- **Smoothing Techniques**: Chen & Goodman, "An Empirical Study of Smoothing Techniques for Language Modeling"
- **Attention Mechanism**: Vaswani et al., "Attention Is All You Need" (2017)

## 👨‍💻 Author

Implemented as part of the AI/ML Intern Assignment.

## 📄 License

This is an educational project. The training corpus (Alice's Adventures in Wonderland) is in the public domain.
