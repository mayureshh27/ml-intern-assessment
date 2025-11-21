"""
Utility functions for text preprocessing in the N-gram language model.

This module provides functions for cleaning, tokenizing, and padding text
for use in the trigram language model.
"""

import re
import string


def clean_text(text):
    """
    Clean and normalize text for training.
    
    Strategy:
    - Convert to lowercase for consistency
    - Preserve sentence-ending punctuation (. ! ?)
    - Remove other punctuation except apostrophes (for contractions)
    - Normalize whitespace
    
    Args:
        text (str): Raw input text
    
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Preserve sentence-ending punctuation by adding space before them
    text = re.sub(r'([.!?])', r' \1', text)
    
    # Remove punctuation except sentence endings and apostrophes
    # Keep: . ! ? '
    # Remove: , ; : " - etc.
    allowed_chars = set(string.ascii_lowercase + string.digits + " .!?'")
    text = ''.join(c if c in allowed_chars else ' ' for c in text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize(text):
    """
    Tokenize text into words.
    
    Args:
        text (str): Cleaned text
    
    Returns:
        list: List of tokens (words)
    """
    # Split on whitespace
    tokens = text.split()
    return tokens


def split_into_sentences(tokens):
    """
    Split a list of tokens into sentences based on sentence-ending punctuation.
    
    Args:
        tokens (list): List of tokens
    
    Returns:
        list: List of sentences, where each sentence is a list of tokens
    """
    sentences = []
    current_sentence = []
    
    sentence_enders = {'.', '!', '?'}
    
    for token in tokens:
        current_sentence.append(token)
        
        # Check if this token ends a sentence
        if token in sentence_enders:
            if current_sentence:  # Only add non-empty sentences
                sentences.append(current_sentence)
                current_sentence = []
    
    # Add any remaining tokens as a sentence
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def add_padding(sentence, start_token='<START>', end_token='<END>', n=3):
    """
    Add start and end padding tokens to a sentence for n-gram modeling.
    
    For a trigram model (n=3), we need 2 start tokens to create the initial context.
    
    Args:
        sentence (list): List of tokens in the sentence
        start_token (str): Token to use for sentence start
        end_token (str): Token to use for sentence end
        n (int): N-gram size (default: 3 for trigram)
    
    Returns:
        list: Padded sentence
    """
    # Add (n-1) start tokens at the beginning
    padding_count = n - 1
    padded = [start_token] * padding_count + sentence + [end_token]
    return padded


def preprocess_text(text, n=3):
    """
    Complete preprocessing pipeline: clean, tokenize, split into sentences, and pad.
    
    Args:
        text (str): Raw input text
        n (int): N-gram size (default: 3 for trigram)
    
    Returns:
        list: List of padded sentences, where each sentence is a list of tokens
    """
    # Clean the text
    cleaned = clean_text(text)
    
    # Tokenize
    tokens = tokenize(cleaned)
    
    # Split into sentences
    sentences = split_into_sentences(tokens)
    
    # Add padding to each sentence
    padded_sentences = [add_padding(sent, n=n) for sent in sentences]
    
    return padded_sentences
