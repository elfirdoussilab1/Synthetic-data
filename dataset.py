# In this file, we will implement the datasets used in our experiments
import torch
import numpy as np
import random
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from model import *
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import utils

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

############ Amazon Review dataset generator ############
class Amazon:
    def __init__(self, n, name) :
        # name is either: books, dvd, elec or kitchen
        self.n = n
        self.name = name
        # Load the dataset
        data = loadmat(f'./data/Amazon/{name}.mat')
        self.X = data['fts'] # shape (N, p)

        # Labels
        self.y = data['labels'].reshape((len(self.X), )).astype(int) # shape (n, ), features are sorted by reversed order (ones then zeros)
        self.y = 1 - 2 * self.y

        # Preprocessing
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        self.vmu_2 = np.mean(self.X[self.y > 0], axis = 0)
        self.vmu_1 = np.mean(self.X[self.y < 0], axis = 0)
        self.vmu = (self.vmu_2 - self.vmu_1) / 2
        self.mu = np.sqrt(abs(np.inner(self.vmu_1 , self.vmu_2)))

        # Train Test split
        self.X_r, self.X_test, self.y_r, self.y_test = train_test_split(self.X, self.y, train_size = n / len(self.y))
    
    def generate_synth_data(self, m, epsilon, rho, phi):
        # Estimate covariance matrix
        p = self.X.shape[1]
        n = self.X_r.shape[0] # total number of available real samples
        vmu_hat = np.mean(self.y_r * self.X_r.T, axis = 1 ) / n

        C = (self.vmu * np.ones((n , p)) ).T
        #C = (vmu_hat * np.ones((n , p)) ).T

        cov = (self.y_r * self.X_r.T - C) @ (self.y_r * self.X_r.T - C).T / n

        # generate synthetic samples
        #X_s, y_s = utils.gaussian_mixture(m, vmu_hat, cov, real = False)
        X_s, y_s = utils.gaussian_mixture(m, self.vmu, cov, real = False)

        # Pruning
        # Noise the labels of the synthetic data
        y_tilde = y_s * (2 * np.random.binomial(size = m, p = 1 - epsilon, n = 1) - 1) # p = P[X = 1], i.e 1 - p = epsilon
        
        # Pruning
        vq = np.zeros(m)
        # Indices of y_tilde = y
        m_1 = (y_tilde == y_s).sum()
        vq[y_tilde == y_s] = np.random.binomial(size = m_1, p = phi, n = 1) 
        vq[y_tilde != y_s] = np.random.binomial(size = m - m_1, p = rho, n = 1)

        return X_s.T, y_s, vmu_hat, vq, y_tilde


############ MNIST dataset generator ############
class MNIST_generator(Dataset):
    def __init__(self, n, m, device, train = True, m_estim = None, estimate_cov = False, supervision = False, threshold = 0.):
        # n is the number of real data per-class !
        # m_estim is the number of synthetic samples PER-CLASS to use to estimate covariance
        # m is the number of synthetic samples to add per-class
        if m_estim is not None:
            assert m >= m_estim
        
        self.device = device

        # Load the train data
        data = datasets.MNIST(root = "data", train = train, download= True, transform= ToTensor())
        y_r = data.targets.cpu().detach().numpy()

        X_r = data.data.cpu().detach().numpy() # (N, 28, 28)
        N = X_r.shape[0]
        X_r = X_r.reshape(N , -1)
        p = X_r.shape[1]
        X_r = X_r.astype(float)

        # If train, select only n per class
        X_real = np.empty((0, p))
        y_real = []
        if n <= 5000 and train:
            for k in range(10):
                X = X_r[y_r == k][:n]
                y = [k] * n

                X_real = np.vstack((X_real, X))
                y_real = y_real + y
            y_real = np.array(y_real)

        else: # Take all samples
            X_real = X_r
            y_real = y_r
        
        # Synthetic dataset
        X_s = np.empty((0, p))
        y_s = []
        if train and m > 0: 
            for k in range(10):
                X = X_real[y_real == k]

                # estimate the mean
                vmu_k = np.mean(X, axis = 0)

                # generate m samples of class k
                X_k_syn = np.random.multivariate_normal(mean = vmu_k, cov = np.eye(p), size = m)

                if estimate_cov:
                    # Take m_estim only
                    X_k_syn = X_k_syn[:m_estim]
                    X = np.vstack((X, X_k_syn)) # shape (n + m_estim, p)

                    # Estimate the mean again
                    vmu_k = np.mean(X, axis = 0)
                    cov_k = (X - vmu_k).T @ (X  -vmu_k)/ (X.shape[0] - 1)
                    Z = np.random.multivariate_normal(mean = vmu_k, cov = cov_k, size = m - m_estim)
                    X_k_syn = np.vstack((X_k_syn, Z)) # shape (m, p)

                # Validate using supervison
                if supervision:
                    # Load Discriminator
                    d_k = Discriminator(in_features=784, out_features=1)
                    state_dict = torch.load(f'./mnist_models/gan-discriminator-mnist-cl-{k}.pth', weights_only= True)
                    d_k.load_state_dict(state_dict)
                    
                    Z = torch.from_numpy(X_k_syn).float()
                    ops = d_k(Z).view(-1) # shape (m,)
                    ops = ops.cpu().detach().numpy() >= threshold
                    # Images to keep are of ops >= 0
                    X_k_syn = X_k_syn[ops] # shape (<m, 784)

                y_k_syn = [k] * len(X_k_syn)

                # Add to the final dataset
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


class MNIST_GAN(Dataset):
    def __init__(self, n, m, device, train = True, supervision = False, threshold = 0.):
        # supervision: using the discrimator as a verifier
        super().__init__()

        self.device = device
        # Load the train data
        data = datasets.MNIST(root = "data", train = train, download= True, transform= ToTensor())
        y_r = data.targets.cpu().detach().numpy()
        X_r = data.data.cpu().detach().numpy() # (N, 28, 28)
        X_r = X_r.reshape(len(y_r), -1) # (N, 28 * 28)
        X_r = X_r.astype(float)
        p = X_r.shape[1]

        # Real data
        # If train, select only n per class
        X_real = np.empty((0, p))
        y_real = []
        if n <= 5000 and train:
            for k in range(10):
                X = X_r[y_r == k][:n]
                y = [k] * len(X)

                X_real = np.vstack((X_real, X))
                y_real = y_real + y
            y_real = np.array(y_real)

        else: # test or n > 5000
            X_real = X_r
            y_real = y_r
        
        # Synthetic dataset
        X_s = np.empty((0, p))
        y_s = []
        if train and m > 0:
            for k in range(10):
                # Load Generator
                g_k = Generator(in_features=784, out_features=784)
                state_dict = torch.load(f'./mnist_models/gan-generator-mnist-cl-{k}.pth', weights_only= True)
                g_k.load_state_dict(state_dict)

                # Load Discriminator
                d_k = Discriminator(in_features=784, out_features=1)
                state_dict = torch.load(f'./mnist_models/gan-discriminator-mnist-cl-{k}.pth', weights_only= True)
                d_k.load_state_dict(state_dict)

                # Generate m samples
                Z = np.random.uniform(-1, 1, size=(m, 784))
                Z = torch.from_numpy(Z).float()
                fake_images = g_k(Z) # shape (m, 784)
                
                if supervision:
                    ops = d_k(fake_images).view(-1) # shape (m,)
                    ops = ops.cpu().detach().numpy() >= threshold
                    # Images to keep are of ops >= 0
                    fake_images = fake_images[ops] # shape (<m, 784)
                fake_images = fake_images.cpu().detach().numpy()

                # Add them to the dataset
                X_s = np.vstack((X_s, fake_images))
                labels = [k] * fake_images.shape[0]
                y_s = y_s + labels

        y_s = np.array(y_s)
        # Separate
        self.X_s = X_s
        self.y_s = y_s
        self.X_real = X_real
        self.y_real = y_real

        # Merged
        self.X = np.vstack((X_real, X_s))
        self.y = np.hstack((y_real, y_s))

        # Shuffle
        idx = np.arange(0, len(self.y))
        random.shuffle(idx)
        self.X = self.X[idx]
        self.y = self.y[idx]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = torch.tensor(self.X[index], dtype = torch.float)
        y = torch.tensor(self.y[index], dtype= torch.long)
        return x.to(self.device), y.to(self.device)

class MNIST_verifier_data(Dataset):
    def __init__(self, m, train, device):
        super().__init__()
        self.device = device

        data = datasets.MNIST(root = "data", train = train, download= True, transform= ToTensor())
        self.X_r = data.data.cpu().detach().numpy() # shape (n, p)
        n, p = self.X_r.shape
        self.y_r = np.ones(n)

        # Generating synthetic samples
        X_s = np.empty((0, p))
        self.y_s = np.zeros(m)

        if m > 0:
            for k in range(10):
                # Load Generator
                g_k = Generator(in_features=784, out_features=784)
                state_dict = torch.load(f'./mnist_models/gan-generator-mnist-cl-{k}.pth', weights_only= True)
                g_k.load_state_dict(state_dict)

                # Generate m samples
                Z = np.random.uniform(-1, 1, size=(m, 784))
                Z = torch.from_numpy(Z).float()
                fake_images = g_k(Z) # shape (m, 784)
                fake_images = fake_images.cpu().detach().numpy()

                # Add them to the dataset
                X_s = np.vstack((X_s, fake_images))

        self.X_s = X_s

        # Merge
        self.X = np.vstack((self.X_r, self.X_s))
        self.y = np.hstack((self.y_r, self.y_s))

        # Shuffle
        idx = np.arange(0, len(self.y))
        random.shuffle(idx)
        self.X = self.X[idx]
        self.y = self.y[idx]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = torch.tensor(self.X[index], dtype = torch.float)
        y = torch.tensor(self.y[index], dtype= torch.long)
        return x.to(self.device), y.to(self.device)
        

