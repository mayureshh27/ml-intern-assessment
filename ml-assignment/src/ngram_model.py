"""
Trigram Language Model Implementation.

This module implements a trigram (N=3) language model from scratch using
nested dictionaries for n-gram storage and probabilistic sampling for text generation.
"""

import random
from collections import defaultdict
import numpy as np
from utils import preprocess_text


class TrigramModel:
    """
    A trigram language model that learns word sequences and generates text.
    
    The model uses nested dictionaries to store trigram counts and implements
    Laplace smoothing to handle unseen n-grams during generation.
    """
    
    def __init__(self, smoothing_alpha=0.01):
        """
        Initializes the TrigramModel.
        
        Args:
            smoothing_alpha (float): Smoothing parameter for Laplace smoothing (default: 0.01)
        """
        # Nested dictionary structure: trigram_counts[word1][word2][word3] = count
        # Using defaultdict for automatic initialization
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Bigram counts for context: bigram_counts[word1][word2] = count
        # Used to normalize probabilities: P(w3|w1,w2) = count(w1,w2,w3) / count(w1,w2)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        
        # Vocabulary: set of all unique words seen during training
        self.vocabulary = set()
        
        # Special tokens
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
        # Smoothing parameter (Laplace/add-alpha smoothing)
        self.alpha = smoothing_alpha
        
        # N-gram size (fixed at 3 for trigram)
        self.n = 3
        
        # Flag to track if model has been trained
        self.is_trained = False

    def fit(self, text):
        """
        Trains the trigram model on the given text.
        
        This method:
        1. Preprocesses the text (cleaning, tokenization, padding)
        2. Extracts all trigrams from the text
        3. Counts trigram and bigram occurrences
        4. Builds the vocabulary
        
        Args:
            text (str): The text to train the model on.
        """
        if not text or not text.strip():
            # Handle empty text
            self.is_trained = True
            return
        
        # Preprocess text: clean, tokenize, split into sentences, and pad
        padded_sentences = preprocess_text(text, n=self.n)
        
        # Extract trigrams and count them
        for sentence in padded_sentences:
            # Add all words (except special tokens) to vocabulary
            for word in sentence:
                if word not in {self.START_TOKEN, self.END_TOKEN}:
                    self.vocabulary.add(word)
            
            # Extract trigrams from this sentence
            for i in range(len(sentence) - 2):
                w1, w2, w3 = sentence[i], sentence[i+1], sentence[i+2]
                
                # Count this trigram
                self.trigram_counts[w1][w2][w3] += 1
                
                # Count the bigram context (w1, w2)
                self.bigram_counts[w1][w2] += 1
        
        # Add special tokens to vocabulary
        self.vocabulary.add(self.START_TOKEN)
        self.vocabulary.add(self.END_TOKEN)
        self.vocabulary.add(self.UNK_TOKEN)
        
        self.is_trained = True

    def generate(self, max_length=50, seed=None):
        """
        Generates new text using the trained trigram model.
        
        This method uses probabilistic sampling to choose the next word based on
        the current bigram context. It converts trigram counts to probabilities
        and samples from this distribution.
        
        Args:
            max_length (int): The maximum length of the generated text (in words).
            seed (int): Random seed for reproducibility (optional).
        
        Returns:
            str: The generated text.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating text. Call fit() first.")
        
        if not self.trigram_counts:
            # Model was trained on empty text
            return ""
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Start with two START tokens as initial context
        context = [self.START_TOKEN, self.START_TOKEN]
        generated_words = []
        
        for _ in range(max_length):
            # Get the current bigram context
            w1, w2 = context[-2], context[-1]
            
            # Get all possible next words given this context
            next_word = self._sample_next_word(w1, w2)
            
            # If we generated an END token, stop
            if next_word == self.END_TOKEN:
                break
            
            # Add the word to our generated sequence
            generated_words.append(next_word)
            
            # Update context (shift window)
            context.append(next_word)
        
        # Join the words and return
        return ' '.join(generated_words)
    
    def _sample_next_word(self, w1, w2):
        """
        Sample the next word given a bigram context using probabilistic sampling.
        
        This method:
        1. Gets all possible next words and their counts
        2. Applies Laplace smoothing
        3. Converts counts to probabilities
        4. Samples from the probability distribution
        
        Args:
            w1 (str): First word of the context
            w2 (str): Second word of the context
        
        Returns:
            str: The sampled next word
        """
        # Get the possible next words and their counts
        if w1 in self.trigram_counts and w2 in self.trigram_counts[w1]:
            next_word_counts = self.trigram_counts[w1][w2]
        else:
            # This bigram context was never seen in training
            # Fall back to uniform distribution over vocabulary
            next_word_counts = {}
        
        if not next_word_counts:
            # No trigrams found for this context
            # Use a simple fallback: sample from common continuations
            # or just return END token to finish the sentence
            return self.END_TOKEN
        
        # Get all possible next words and their counts
        words = list(next_word_counts.keys())
        counts = np.array([next_word_counts[w] for w in words], dtype=float)
        
        # Apply Laplace smoothing: add alpha to all counts
        # This ensures no word has zero probability
        smoothed_counts = counts + self.alpha
        
        # Convert to probabilities (normalize)
        probabilities = smoothed_counts / smoothed_counts.sum()
        
        # Sample from the probability distribution
        next_word = np.random.choice(words, p=probabilities)
        
        return next_word
    
    def get_trigram_probability(self, w1, w2, w3):
        """
        Calculate the probability of a trigram P(w3|w1,w2).
        
        This is useful for evaluation and debugging.
        
        Args:
            w1 (str): First word
            w2 (str): Second word
            w3 (str): Third word
        
        Returns:
            float: Probability of w3 given (w1, w2)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        
        # Get the count of this trigram
        trigram_count = self.trigram_counts.get(w1, {}).get(w2, {}).get(w3, 0)
        
        # Get the count of the bigram context
        bigram_count = self.bigram_counts.get(w1, {}).get(w2, 0)
        
        if bigram_count == 0:
            return 0.0
        
        # Apply smoothing
        vocab_size = len(self.vocabulary)
        smoothed_prob = (trigram_count + self.alpha) / (bigram_count + self.alpha * vocab_size)
        
        return smoothed_prob

