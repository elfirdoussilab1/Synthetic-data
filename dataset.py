# In this file, we will implement the datasets used in our experiments
import torch
import numpy as np
import random
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

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

############ MNIST dataset generator ############
class MNIST_generator(Dataset):
    def __init__(self, n, m, device, train = True, m_estim = None, estimate_cov = False):
        # n is the number of real data per-class !
        # m_estim is the number of synthetic samples PER-CLASS to use to estimate covariance
        # m is the number of synthetic samples to add per-class
        if m_estim is not None:
            assert m > m_estim
        
        self.m = m
        self.device = device

        # Load the train data
        data = datasets.MNIST(root = "data", train = train, download= True, transform= ToTensor())
        y_r = data.targets.cpu().detach().numpy()

        X_r = data.data.cpu().detach().numpy() # (N, 28, 28)
        X_r = X_r.reshape(X_r.shape[0], -1)
        p = X_r.shape[1]
        X_r = X_r.astype(float)

        # If train, select only n per class
        X_real = np.empty((0, p))
        y_real = []
        if train:
            for k in range(10):
                X = X_r[y_r == k][:n]
                y = [k] * n

                X_real = np.vstack((X_real, X))
                y_real = y_real + y
        
        # Synthetic dataset
        X_s = np.empty((0, p))
        y_s = []
        if train: 
            for k in range(10):
                X = X_real[y_real == k]

                # estimate the mean
                vmu_k = np.mean(X, axis = 0)

                # generate m samples of class k
                X_k_syn = np.random.multivariate_normal(mean = vmu_k, cov = np.eye(p), size = m)
                y_k_syn = [k] * len(X_k_syn)

                if estimate_cov:
                    # Take m_estim only
                    X_k_syn = X_k_syn[:m_estim]
                    X = np.vstack((X, X_k_syn)) # shape (n + m_estim, p)

                    # Estimate the mean again
                    vmu_k = np.mean(X, axis = 0)
                    cov_k = (X - vmu_k).T @ (X  -vmu_k)/ (X.shape[0] - 1)
                    Z = np.random.multivariate_normal(mean = vmu_k, cov = cov_k, size = m - m_estim)
                    X_k_syn = np.vstack((X_k_syn, Z)) # shaoe (m, p)

                # Add to the final dataset
                assert X_k_syn.shape[0] == m
                X_s = np.vstack((X_s, X_k_syn))
                y_s = y_s + y_k_syn

        y_s = np.array(y_s)
        # Separate
        self.X_s = X_s
        self.y_s = y_s
        self.X_real = X_real
        self.y_real = y_real

        # Merged
        self.X = np.vstack((X_real, X_s))
        self.y = np.hstack((y_real, y_s))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = torch.tensor(self.X[index], dtype = torch.float)
        y = torch.tensor(self.y[index], dtype= torch.long)
        return x.to(self.device), y.to(self.device)

