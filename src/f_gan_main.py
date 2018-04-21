import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data  import DataLoader
import torch
from torch.autograd import Variable
from gmm import GMM
import matplotlib.pyplot as plt
from itertools import islice


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,2)
    def forward(self, x):
        output = F.relu(self.fc2(F.relu(self.fc1(x))))
        return output

class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        output = F.relu(self.fc2(F.relu(self.fc1(x))))
        output = output.mean(0)
        return output

n_samples = 1000
n_epochs = 200
batch_size = 1

gmm = GMM(n_gaussians=2, dim=2, random_seed=100)
sample_data = torch.Tensor(gmm.sample(n_samples))
print(sample_data.size())
dataloader_train = DataLoader(sample_data, batch_size=batch_size)


noise = torch.FloatTensor(n_samples, 2).normal_(0, 1)
fixed_noise = torch.FloatTensor(n_samples, 2).normal_(0, 1)

netG = Generator()
netD = Discrimator()

optim_netD = torch.optim.Adam(netD.parameters(), lr = 0.01)
optim_netG = torch.optim.Adam(netG.parameters(), lr = 0.01)
