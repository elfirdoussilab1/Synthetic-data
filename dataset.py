# In this file, we will implement the datasets used in our experiments
import torch
import numpy as np
import random

# Generate synthetic data with hierarchical structure
class SimpleGrammar:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.noun_range = (1, vocab_size // 5)
        self.verb_range = (vocab_size // 5, 2 * vocab_size // 5)
        self.adj_range = (2 * vocab_size // 5, 3 * vocab_size // 5)
        self.adv_range = (3 * vocab_size // 5, 4 * vocab_size // 5)
        self.conj_range = (4 * vocab_size // 5, vocab_size)
        self.scheme = {self.noun_range : 'noun', self.verb_range: 'verb', self.adj_range: 'adj',
                       self.adv_range: 'adv',  self.conj_range: 'conj'}
        
    def generate_sentence(self):
        sentence = []
        sentence.extend(self.generate_noun_phrase())
        sentence.extend(self.generate_verb_phrase())
        return sentence
 
    def generate_noun_phrase(self):
        if random.random() < 0.5:
            return [random.randint(*self.noun_range)]
        else:
            return [random.randint(*self.adj_range), random.randint(*self.noun_range)]
 
    def generate_verb_phrase(self):
        if random.random() < 0.3:
            return [random.randint(*self.verb_range)]
        elif random.random() < 0.6:
            return [random.randint(*self.verb_range), random.randint(*self.adv_range)]
        else:
            return [random.randint(*self.verb_range), *self.generate_noun_phrase()]
    
    def seq_scheme(self, sentence):
        l = []
        for word in sentence:
            for word_range, name in self.scheme.items():
                if word_range[0] <= word < word_range[1]:
                    l.append(name)
        return l
 
def generate_grammar_data(num_samples, grammar, max_length):
    data = []
    for _ in range(num_samples):
        sentence = []
        while len(sentence) < max_length:
            sentence.extend(grammar.generate_sentence())
            if len(sentence) < max_length - 1 and random.random() < 0.3:
                sentence.append(random.randint(*grammar.conj_range))
        sentence = sentence[:max_length]
        # Ensure all indices are within range
        sentence = [min(token, grammar.vocab_size - 1) for token in sentence]
        data.append(sentence)
    return torch.LongTensor(data)