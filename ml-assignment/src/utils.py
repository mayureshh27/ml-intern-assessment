import re
import string


def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'([.!?])', r' \1', text)
    allowed_chars = set(string.ascii_lowercase + string.digits + " .!?'")
    text = ''.join(c if c in allowed_chars else ' ' for c in text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    tokens = text.split()
    return tokens


def split_into_sentences(tokens):
    sentences = []
    current_sentence = []
    
    sentence_enders = {'.', '!', '?'}
    
    for token in tokens:
        current_sentence.append(token)
        
        if token in sentence_enders:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def add_padding(sentence, start_token='<START>', end_token='<END>', n=3):
    padding_count = n - 1
    padded = [start_token] * padding_count + sentence + [end_token]
    return padded


def preprocess_text(text, n=3):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    sentences = split_into_sentences(tokens)
    padded_sentences = [add_padding(sent, n=n) for sent in sentences]
    return padded_sentences
