<<<<<<< HEAD
"""
Comprehensive test suite for the Trigram Language Model.

Tests cover model initialization, training, generation, edge cases,
and probability calculations.
"""

import pytest
import numpy as np
from src.ngram_model import TrigramModel


def test_fit_and_generate():
    """Test basic fit and generate functionality."""
=======
import pytest
from src.ngram_model import TrigramModel

def test_fit_and_generate():
>>>>>>> cfff0cadc0242e1f10eec701b355ee9b26226a11
    model = TrigramModel()
    text = "I am a test sentence. This is another test sentence."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)
    assert len(generated_text.split()) > 0

<<<<<<< HEAD

def test_empty_text():
    """Test that empty text is handled gracefully."""
=======
def test_empty_text():
>>>>>>> cfff0cadc0242e1f10eec701b355ee9b26226a11
    model = TrigramModel()
    text = ""
    model.fit(text)
    generated_text = model.generate()
    assert generated_text == ""

<<<<<<< HEAD

def test_short_text():
    """Test that short text is handled correctly."""
=======
def test_short_text():
>>>>>>> cfff0cadc0242e1f10eec701b355ee9b26226a11
    model = TrigramModel()
    text = "I am."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)


<<<<<<< HEAD
def test_trigram_counting():
    """Test that trigrams are counted correctly."""
    model = TrigramModel()
    text = "the cat sat. the cat ran."
    model.fit(text)
    
    # Check that we have trigram counts
    assert len(model.trigram_counts) > 0
    
    # Check that specific trigrams exist
    # After preprocessing: "<START> <START> the", "<START> the cat", "the cat sat", etc.
    assert model.START_TOKEN in model.trigram_counts
    assert 'the' in model.trigram_counts[model.START_TOKEN][model.START_TOKEN]


def test_probability_normalization():
    """Test that probabilities sum to approximately 1.0."""
    model = TrigramModel()
    text = "the cat sat on the mat. the dog ran in the park."
    model.fit(text)
    
    # Get a context that has multiple possible next words
    w1, w2 = model.START_TOKEN, model.START_TOKEN
    
    if w1 in model.trigram_counts and w2 in model.trigram_counts[w1]:
        next_word_counts = model.trigram_counts[w1][w2]
        
        # Calculate probabilities manually
        counts = np.array(list(next_word_counts.values()), dtype=float)
        smoothed_counts = counts + model.alpha
        probabilities = smoothed_counts / smoothed_counts.sum()
        
        # Check that probabilities sum to 1.0 (within floating point tolerance)
        assert abs(probabilities.sum() - 1.0) < 1e-6


def test_vocabulary_building():
    """Test that vocabulary is built correctly."""
    model = TrigramModel()
    text = "hello world. hello there."
    model.fit(text)
    
    # Check that vocabulary contains expected words
    assert 'hello' in model.vocabulary
    assert 'world' in model.vocabulary
    assert 'there' in model.vocabulary
    
    # Check that special tokens are in vocabulary
    assert model.START_TOKEN in model.vocabulary
    assert model.END_TOKEN in model.vocabulary
    assert model.UNK_TOKEN in model.vocabulary


def test_sentence_boundary_handling():
    """Test that sentence boundaries are handled correctly."""
    model = TrigramModel()
    text = "First sentence. Second sentence."
    model.fit(text)
    
    # Each sentence should start with START tokens
    assert model.START_TOKEN in model.trigram_counts
    
    # Each sentence should end with END token
    # Check that some trigram ends with END_TOKEN
    has_end_token = False
    for w1_dict in model.trigram_counts.values():
        for w2_dict in w1_dict.values():
            if model.END_TOKEN in w2_dict:
                has_end_token = True
                break
        if has_end_token:
            break
    
    assert has_end_token, "No trigrams ending with END_TOKEN found"


def test_deterministic_generation_with_seed():
    """Test that generation is deterministic when using a seed."""
    model = TrigramModel()
    text = "the quick brown fox jumps over the lazy dog. " * 10
    model.fit(text)
    
    # Generate with same seed twice
    text1 = model.generate(max_length=20, seed=42)
    text2 = model.generate(max_length=20, seed=42)
    
    assert text1 == text2, "Generation should be deterministic with same seed"


def test_max_length_respected():
    """Test that generated text respects max_length parameter."""
    model = TrigramModel()
    text = "word " * 100  # Lots of repetition
    model.fit(text)
    
    max_len = 10
    generated = model.generate(max_length=max_len)
    word_count = len(generated.split())
    
    # Should be <= max_length (might be less if END token is generated)
    assert word_count <= max_len


def test_model_not_trained_error():
    """Test that generating before training raises an error."""
    model = TrigramModel()
    
    with pytest.raises(ValueError, match="Model must be trained"):
        model.generate()


def test_get_trigram_probability():
    """Test the trigram probability calculation method."""
    model = TrigramModel()
    text = "the cat sat. the cat sat. the cat sat."  # Repeated for higher counts
    model.fit(text)
    
    # Get probability of a trigram that should exist
    prob = model.get_trigram_probability(model.START_TOKEN, model.START_TOKEN, 'the')
    
    # Probability should be between 0 and 1
    assert 0.0 <= prob <= 1.0
    
    # Probability should be > 0 for a trigram that exists
    assert prob > 0.0


def test_large_corpus_training():
    """Test that model can handle larger corpus."""
    model = TrigramModel()
    
    # Create a larger synthetic corpus
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Actions speak louder than words."
    ] * 20  # Repeat to make it larger
    
    text = " ".join(sentences)
    model.fit(text)
    
    # Should have a reasonable vocabulary size
    assert len(model.vocabulary) > 10
    
    # Should be able to generate text
    generated = model.generate(max_length=30)
    assert isinstance(generated, str)
    assert len(generated) > 0



=======
>>>>>>> cfff0cadc0242e1f10eec701b355ee9b26226a11
