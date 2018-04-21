import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data  import DataLoader
import torch
from torch.autograd import Variable
from gmm import GMM
import matplotlib.pyplot as plt
from itertools import islice

#BX2
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

gmm = GMM(n_gaussians=2, dim=2, random_seed=100)

discrimator_iters = 100
generator_iters = 25
n_samples = 1000
n_epochs = 200
batch_size = 1
sample_data = torch.Tensor(gmm.sample(n_samples))
print(sample_data.size())
dataloader_train = DataLoader(sample_data, batch_size=batch_size)
one = torch.FloatTensor([1])
neg_one = one * -1
noise = torch.FloatTensor(n_samples, 2).normal_(0, 1)
fixed_noise = torch.FloatTensor(n_samples, 2).normal_(0, 1)



netG = Generator()
netD = Discrimator()

optim_netD = torch.optim.Adam(netD.parameters(), lr = 0.001)
optim_netG = torch.optim.Adam(netG.parameters(), lr = 0.001)


for _ in range(n_epochs):
    net_g_end = 0
    dataloader_iter = iter(dataloader_train)
    total_counter = 0

    while total_counter < len(dataloader_iter):
        netD_counter = 0
        netG_counter = 0

        while netD_counter < discrimator_iters:
            batch_real = next(dataloader_iter)
            total_counter += 1

            #Lipshitz constraint
            for params in netD.parameters():
                    params.data.clamp_(-0.1, 0.01)

            netD.zero_grad()
            d_error_real = netD(Variable(batch_real))
            d_error_real.backward(one)

            noise.resize_(batch_size, 2).normal_(0, 1)
            with torch.no_grad():
                batch_fake = netG(Variable(noise))

            d_error_fake = netD(batch_fake)
            d_error_fake.backward(neg_one)

            d_error_total = d_error_real + d_error_fake

            optim_netD.step()
            netD_counter += 1

        while netG_counter < generator_iters:
            netG.zero_grad()
            noise.resize_(batch_size, 2).normal_(0, 1)
            fake = netG(Variable(noise))
            g_error = netD(fake)

            g_error.backward(one)

            optim_netG.step()
            netG_counter += 1


with torch.no_grad():
    gen_data = netG(Variable(fixed_noise)).data

plt.figure(1)
plt.subplot(211)

plt.scatter(sample_data[:,0], sample_data[:, 1])
means = gmm.means.T
plt.scatter(means[:,0], means[:,1], c='red')

plt.subplot(212)
print(gen_data.size())
plt.scatter(gen_data[:,0], gen_data[:, 1])
plt.show()
