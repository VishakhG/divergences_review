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
        output = F.sigmoid(self.fc2(F.relu(self.fc1(x))))

        return output

class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        output = F.relu(self.fc2(F.relu(self.fc1(x))))
        return output

n_samples = 1000
n_epochs = 200
batch_size = 1

gmm = GMM(n_gaussians=2, dim=2, random_seed=100)
sample_data = torch.Tensor(gmm.sample(n_samples))

dataloader_train = DataLoader(sample_data, batch_size=batch_size)


noise = torch.FloatTensor(n_samples, 2).normal_(0, 1)
fixed_noise = torch.FloatTensor(n_samples, 2).normal_(0, 1)

netG = Generator()
netD = Discrimator()

optim_netD = torch.optim.Adam(netD.parameters(), lr = 0.01)
optim_netG = torch.optim.Adam(netG.parameters(), lr = 0.01)


def discrimator_loss(D_real, D_fake, divergence_choice='forward_kl'):
    if divergence_choice == 'forward_kl':
        return  -(torch.mean(D_real) - torch.mean(torch.exp(D_fake - 1)))

    elif divergence_choice == 'total_variation':
        return -(torch.mean(0.5 * torch.tanh(D_real)) -
                          torch.mean(0.5 * torch.tanh(D_fake)))

    elif divergence_choice == 'pearson_chi_squared':
        return -(torch.mean(D_real) - torch.mean(0.25*D_fake**2 + D_fake))

    elif divergence_choice == 'squared_hellinger':
        return  -(torch.mean(1 - torch.exp(D_real)) -
                  torch.mean((1 - torch.exp(D_fake)) / (torch.exp(D_fake))))


def generator_loss(D_fake, divergence_choice='forward_kl'):
    if divergence_choice == 'total_variation':
        return  -torch.mean(0.5 * torch.tanh(D_fake))

    elif divergence_choice == 'forward_kl':
        return -torch.mean(torch.exp(D_fake - 1))

    elif divergence_choice == 'reverse_kl':
        return -torch.mean(-1 - D_fake)
    elif divergence_choice == 'pearson_chi_squared':
        return -torch.mean(0.25*D_fake**2 + D_fake)

    elif divergence_choice == 'squared_hellinger':
        return -torch.mean((1 - torch.exp(D_fake)) / (torch.exp(D_fake)))



for i in range(n_epochs):
    for _, batch_real in enumerate(dataloader_train):
        z = Variable(torch.randn(batch_size, 2))
        batch_real = Variable(batch_real)
        batch_fake = netG(z)

        netD.reset_grad()
        d_real = netD(batch_real)
        d_fake = netD(batch_fake)

        d_loss = discrimator_loss(d_real, d_fake)
        d_loss.backward()
        optim_netD.step()

        netG.reset_grad()
        batch_fake = netG(z)
        d_fake = netD(batch_fake)

        g_loss = generator_loss(D_fake)
        g_loss.backward()
        optim_netG.step()

