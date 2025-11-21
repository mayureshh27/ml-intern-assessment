import random
from collections import defaultdict
import numpy as np
from .utils import preprocess_text


class TrigramModel:
    
    def __init__(self, smoothing_alpha=0.01):
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        self.alpha = smoothing_alpha
        self.n = 3
        self.is_trained = False

    def fit(self, text):
        if not text or not text.strip():
            self.is_trained = True
            return
        
        padded_sentences = preprocess_text(text, n=self.n)
        
        for sentence in padded_sentences:
            for word in sentence:
                if word not in {self.START_TOKEN, self.END_TOKEN}:
                    self.vocabulary.add(word)
            
            for i in range(len(sentence) - 2):
                w1, w2, w3 = sentence[i], sentence[i+1], sentence[i+2]
                self.trigram_counts[w1][w2][w3] += 1
                self.bigram_counts[w1][w2] += 1
        
        self.vocabulary.add(self.START_TOKEN)
        self.vocabulary.add(self.END_TOKEN)
        self.vocabulary.add(self.UNK_TOKEN)
        
        self.is_trained = True

    def generate(self, max_length=50, seed=None):
        if not self.is_trained:
            raise ValueError("Model must be trained before generating text. Call fit() first.")
        
        if not self.trigram_counts:
            return ""
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        context = [self.START_TOKEN, self.START_TOKEN]
        generated_words = []
        
        for _ in range(max_length):
            w1, w2 = context[-2], context[-1]
            next_word = self._sample_next_word(w1, w2)
            
            if next_word == self.END_TOKEN:
                break
            
            generated_words.append(next_word)
            context.append(next_word)
        
        return ' '.join(generated_words)
    
    def _sample_next_word(self, w1, w2):
        if w1 in self.trigram_counts and w2 in self.trigram_counts[w1]:
            next_word_counts = self.trigram_counts[w1][w2]
        else:
            next_word_counts = {}
        
        if not next_word_counts:
            return self.END_TOKEN
        
        words = list(next_word_counts.keys())
        counts = np.array([next_word_counts[w] for w in words], dtype=float)
        smoothed_counts = counts + self.alpha
        probabilities = smoothed_counts / smoothed_counts.sum()
        next_word = np.random.choice(words, p=probabilities)
        
        return next_word
    
    def get_trigram_probability(self, w1, w2, w3):
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        
        trigram_count = self.trigram_counts.get(w1, {}).get(w2, {}).get(w3, 0)
        bigram_count = self.bigram_counts.get(w1, {}).get(w2, 0)
        
        if bigram_count == 0:
            return 0.0
        
        vocab_size = len(self.vocabulary)
        smoothed_prob = (trigram_count + self.alpha) / (bigram_count + self.alpha * vocab_size)
        
        return smoothed_prob
    
    def calculate_perplexity(self, text):
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        
        if not text or not text.strip():
            return float('inf')
        
        padded_sentences = preprocess_text(text, n=self.n)
        
        total_log_prob = 0.0
        total_trigrams = 0
        
        for sentence in padded_sentences:
            for i in range(len(sentence) - 2):
                w1, w2, w3 = sentence[i], sentence[i+1], sentence[i+2]
                
                prob = self.get_trigram_probability(w1, w2, w3)
                
                if prob > 0:
                    total_log_prob += np.log(prob)
                else:
                    total_log_prob += np.log(1e-10)
                
                total_trigrams += 1
        
        if total_trigrams == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_trigrams
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity

