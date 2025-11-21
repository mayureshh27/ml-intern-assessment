import pytest
import numpy as np
from src.ngram_model import TrigramModel


def test_fit_and_generate():
    model = TrigramModel()
    text = "I am a test sentence. This is another test sentence."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)
    assert len(generated_text.split()) > 0

def test_empty_text():
    model = TrigramModel()
    text = ""
    model.fit(text)
    generated_text = model.generate()
    assert generated_text == ""


def test_short_text():
    model = TrigramModel()
    text = "I am."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)


def test_trigram_counting():
    model = TrigramModel()
    text = "the cat sat. the cat ran."
    model.fit(text)
    
    assert len(model.trigram_counts) > 0
    assert model.START_TOKEN in model.trigram_counts
    assert 'the' in model.trigram_counts[model.START_TOKEN][model.START_TOKEN]


def test_probability_normalization():
    model = TrigramModel()
    text = "the cat sat on the mat. the dog ran in the park."
    model.fit(text)
    
    w1, w2 = model.START_TOKEN, model.START_TOKEN
    
    if w1 in model.trigram_counts and w2 in model.trigram_counts[w1]:
        next_word_counts = model.trigram_counts[w1][w2]
        
        counts = np.array(list(next_word_counts.values()), dtype=float)
        smoothed_counts = counts + model.alpha
        probabilities = smoothed_counts / smoothed_counts.sum()
        
        assert abs(probabilities.sum() - 1.0) < 1e-6


def test_vocabulary_building():
    model = TrigramModel()
    text = "hello world. hello there."
    model.fit(text)
    
    assert 'hello' in model.vocabulary
    assert 'world' in model.vocabulary
    assert 'there' in model.vocabulary
    
    assert model.START_TOKEN in model.vocabulary
    assert model.END_TOKEN in model.vocabulary
    assert model.UNK_TOKEN in model.vocabulary


def test_sentence_boundary_handling():
    model = TrigramModel()
    text = "First sentence. Second sentence."
    model.fit(text)
    
    assert model.START_TOKEN in model.trigram_counts
    
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
    model = TrigramModel()
    text = "the quick brown fox jumps over the lazy dog. " * 10
    model.fit(text)
    
    text1 = model.generate(max_length=20, seed=42)
    text2 = model.generate(max_length=20, seed=42)
    
    assert text1 == text2, "Generation should be deterministic with same seed"


def test_max_length_respected():
    model = TrigramModel()
    text = "word " * 100
    model.fit(text)
    
    max_len = 10
    generated = model.generate(max_length=max_len)
    word_count = len(generated.split())
    
    assert word_count <= max_len


def test_model_not_trained_error():
    model = TrigramModel()
    
    with pytest.raises(ValueError, match="Model must be trained"):
        model.generate()


def test_get_trigram_probability():
    model = TrigramModel()
    text = "the cat sat. the cat sat. the cat sat."
    model.fit(text)
    
    prob = model.get_trigram_probability(model.START_TOKEN, model.START_TOKEN, 'the')
    
    assert 0.0 <= prob <= 1.0
    assert prob > 0.0


def test_large_corpus_training():
    model = TrigramModel()
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Actions speak louder than words."
    ] * 20
    
    text = " ".join(sentences)
    model.fit(text)
    
    assert len(model.vocabulary) > 10
    
    generated = model.generate(max_length=30)
    assert isinstance(generated, str)
    assert len(generated) > 0


def test_calculate_perplexity_basic():
    model = TrigramModel()
    text = "the cat sat on the mat. the dog ran in the park."
    model.fit(text)
    
    perplexity = model.calculate_perplexity(text)
    
    assert isinstance(perplexity, (float, np.floating))
    assert perplexity > 0
    assert not np.isnan(perplexity)
    assert not np.isinf(perplexity)


def test_perplexity_on_training_data():
    model = TrigramModel()
    text = "the quick brown fox jumps over the lazy dog. " * 5
    model.fit(text)
    
    perplexity = model.calculate_perplexity(text)
    
    assert perplexity > 0
    assert perplexity < 1000


def test_perplexity_unseen_text():
    model = TrigramModel()
    train_text = "the cat sat on the mat."
    test_text = "a dog ran in the park."
    
    model.fit(train_text)
    perplexity = model.calculate_perplexity(test_text)
    
    assert perplexity > 0


def test_perplexity_empty_text():
    model = TrigramModel()
    text = "the cat sat."
    model.fit(text)
    
    perplexity = model.calculate_perplexity("")
    
    assert np.isinf(perplexity)


def test_perplexity_better_on_similar_text():
    model = TrigramModel()
    train_text = "the cat sat. the cat ran. the cat jumped."
    similar_text = "the cat walked."
    different_text = "a dog barked loudly."
    
    model.fit(train_text)
    
    perplexity_similar = model.calculate_perplexity(similar_text)
    perplexity_different = model.calculate_perplexity(different_text)
    
    assert perplexity_similar < perplexity_different

