#AI/ML Intern Assignment

Welcome to the AI/ML Intern assignment! This project is designed to test your core Python skills and your ability to design and build a clean and efficient system from scratch.

## ✅ Completed Tasks

- **Task 1**: Trigram Language Model (Required) ✓
- **Task 2**: Scaled Dot-Product Attention (Optional) ✓

## Quick Start

### 1. Set Up Virtual Environment

```bash
# Clone or navigate to the repository
cd ml-intern-assessment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Git Bash/MinGW):
./venv/Scripts/activate
# On Windows (CMD):
venv\Scripts\activate
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

This downloads "Alice's Adventures in Wonderland" from Project Gutenberg.

### 3. Run the Trigram Model

```bash
# Generate text samples
python generate.py
```

### 4. Run Tests (Optional)

```bash
cd ../..
python -m pytest ml-assignment/tests/test_ngram.py -v
```

### 5. Try the Attention Mechanism (Optional)

```bash
cd ml-assignment/attention
python demo.py
```

## 📁 Project Structure

```
ml-intern-assessment/
├── ml-assignment/
│   ├── src/                 # Source code
│   │   ├── ngram_model.py   # Trigram model implementation
│   │   ├── utils.py         # Text preprocessing
│   │   ├── generate.py      # Text generation script
│   │   └── download_corpus.py
│   ├── tests/               # Test suite
│   │   └── test_ngram.py
│   ├── attention/           # Optional Task 2
│   │   ├── scaled_attention.py
│   │   └── demo.py
│   ├── data/                # Training corpus
│   ├── evaluation.md        # Design decisions (1-page summary)
│   └── README.md            # Detailed documentation
├── requirements.txt         # Dependencies
├── assignment.md            # Assignment description
└── quick_start.md          # Quick reference
```

## 📝 Documentation

- **[ml-assignment/README.md](ml-assignment/README.md)**: Comprehensive guide with usage examples
- **[ml-assignment/evaluation.md](ml-assignment/evaluation.md)**: Design decisions and analysis
- **[assignment.md](assignment.md)**: Original assignment requirements
- **[quick_start.md](quick_start.md)**: Quick reference checklist

## 🎯 Implementation Highlights

### Task 1: Trigram Language Model
- ✅ Nested dictionary storage for efficient n-gram lookup
- ✅ Laplace smoothing for unknown word handling
- ✅ Probabilistic sampling for diverse text generation
- ✅ Sentence-level padding with special tokens
- ✅ Comprehensive test suite (13 tests)
- ✅ Trained on Alice in Wonderland (~26K words, 2.6K vocabulary)

### Task 2: Scaled Dot-Product Attention (Optional)
- ✅ NumPy-only implementation
- ✅ Numerically stable softmax
- ✅ Causal masking support
- ✅ Comprehensive demonstrations
- ✅ Detailed mathematical explanations

## 📊 Results

### Model Statistics
- **Vocabulary**: 2,583 unique words
- **Trigrams**: 23,603 unique trigrams
- **Training time**: <1 second
- **Generation quality**: Grammatically plausible, stylistically consistent

### Sample Generated Text
```
"and certainly there was generally a frog and both the hedgehogs were out of sight"
"alice waited patiently ."
"the queen to day !"
```

## 🔧 Requirements

- Python 3.7+
- NumPy
- Pytest
- Requests

All dependencies are listed in `requirements.txt`.

## 🐛 Troubleshooting

### Virtual Environment Issues
Make sure to activate the virtual environment before running any scripts:
```bash
./venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Corpus Download Fails
If network issues prevent automatic download:
1. Visit https://www.gutenberg.org/ebooks/11
2. Download as plain text
3. Save to `ml-assignment/data/alice_in_wonderland.txt`

### Import Errors
Ensure you're in the correct directory and dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📚 Additional Resources

- Project Gutenberg: https://www.gutenberg.org
- N-gram Language Models: Jurafsky & Martin textbook
- Attention Mechanism: "Attention Is All You Need" paper

## 👨‍💻 Development Notes

This implementation prioritizes:
- **Code clarity**: Well-documented, easy to understand
- **Best practices**: Proper error handling, type hints, docstrings
- **Efficiency**: Appropriate data structures for the task
- **Testability**: Comprehensive test coverage

For detailed design rationale, see [evaluation.md](ml-assignment/evaluation.md).

