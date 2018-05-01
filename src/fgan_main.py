import matplotlib
matplotlib.use('Agg')
import numpy as np # noqa
import seaborn as sns # noqa
import matplotlib.pyplot as plt # noqa
import torch.nn as nn # noqa
import torch.nn.functional as F # noqa
from torch.utils.data  import DataLoader # noqa
import torch # noqa
from torch.autograd import Variable # noqa
from gmm import GMM # noqa
import matplotlib.pyplot as plt # noqa
from itertools import islice # noqa
from utils import plot_density_data # noqa
from args import get_args # noqa
from utils import Generator, pdf_normal # noqa

args = get_args('FGAN')


class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        output = F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x))))
        return output


n_samples = args.n_data
n_epochs = args.n_epochs
batch_size = args.batch_size

gmm = GMM(n_gaussians=2, dim=2, random_seed=22)
sample_data = torch.Tensor(gmm.sample(n_samples))

dataloader_train = DataLoader(sample_data, batch_size=batch_size)


noise = torch.randn(batch_size, 2)
fixed_noise = torch.randn(n_samples, 2)

netG = Generator(sample_data.numpy())
netD = Discrimator()

optim_netD = torch.optim.Adam(netD.parameters(), lr=args.lr)
optim_netG = torch.optim.Adam(netG.parameters(), lr=args.lr)

if torch.cuda.is_available():
    netG = netG.cuda()
    netD = netD.cuda()
    sample_data = sample_data.cuda()
    noise = noise.cuda()
    fixed_noise = fixed_noise.cuda()

divergence_choice = args.divergence_type


print("Using divergence choice {0}".format(divergence_choice))


def discrimator_loss(D_real, D_fake, divergence_choice='FORWARD_KL'):
    if divergence_choice == 'FORWARD_KL':
        return -(torch.mean(D_real) - torch.mean(torch.exp(D_fake - 1)))
    elif divergence_choice == 'REVERSE_KL':
        return -(torch.mean(-torch.exp(D_real)) - torch.mean(-1 - D_fake))

    elif divergence_choice == 'TOTAL_VARIATION':
        return -(torch.mean(0.5 * torch.tanh(D_real)) -
                 torch.mean(0.5 * torch.tanh(D_fake)))

    elif divergence_choice == 'PEARSON_CHI_SQUARED':
        return -(torch.mean(D_real) - torch.mean(0.25*D_fake**2 + D_fake))

    elif divergence_choice == 'SQUARED_HELLINGER':
        return -(torch.mean(1 - torch.exp(D_real)) -
                 torch.mean((1 - torch.exp(D_fake)) / (torch.exp(D_fake))))


def generator_loss(D_fake, divergence_choice='FORWARD_KL'):
    if divergence_choice == 'TOTAL_VARIATION':
        return -torch.mean(0.5 * torch.tanh(D_fake))

    elif divergence_choice == 'FORWARD_KL':
        return -torch.mean(torch.exp(D_fake - 1))

    elif divergence_choice == 'REVERSE_KL':
        return -torch.mean(-1 - D_fake)
    elif divergence_choice == 'PEARSON_CHI_SQUARED':
        return -torch.mean(0.25*D_fake**2 + D_fake)

    elif divergence_choice == 'SQUARED_HELLINGER':
        return -torch.mean((1 - torch.exp(D_fake)) / (torch.exp(D_fake)))


for epoch_i in range(n_epochs):
    epoch_discriminator_losses = []
    epoch_generator_losses = []

    for _, batch_real in enumerate(dataloader_train):
        z = Variable(torch.randn(batch_size, 2))
        batch_real = Variable(batch_real, requires_grad=True)
        if torch.cuda.is_available():
            batch_real = batch_real.cuda()
            z = z.cuda()
        batch_fake = netG(z)

        optim_netD.zero_grad()
        d_real = netD(batch_real)
        d_fake = netD(batch_fake)

        d_loss = discrimator_loss(d_real, d_fake,
                                  divergence_choice=divergence_choice)
        epoch_discriminator_losses.append(d_loss.data[0])
        d_loss.backward()
        optim_netD.step()

        optim_netG.zero_grad()
        batch_fake = netG(z)
        d_fake = netD(batch_fake)

        g_loss = generator_loss(d_fake, divergence_choice=divergence_choice)
        epoch_generator_losses.append(g_loss.data[0])
        g_loss.backward()
        optim_netG.step()

    if epoch_i % args.save_freq == 0:
        print("Batch {0} generator loss {1} discriminator_loss {2}".format(
            epoch_i, np.mean(epoch_discriminator_losses),
            np.mean(epoch_generator_losses)))

if torch.cuda.is_available():
    gen_data = (netG(Variable(fixed_noise, volatile=True))).cpu().data.numpy()
else:
    with torch.no_grad():
        gen_data = (netG(Variable(fixed_noise))).data.numpy()


if torch.cuda.is_available():
    sample_data = sample_data.cpu()

sample_data = sample_data.cpu().numpy()
mu, std = netG.get_params()
mu = mu.cpu().data.numpy()
std = std.cpu().data.numpy()
std = np.fill_diagonal(np.zeros((mu.shape[0], mu.shape[0])), std[0])
pdf = pdf_normal(mu, std)
plot_density_data(pdf, sample_data)
plt.savefig('../plots/F-gan samples with divergence ' +
            (args.divergence_type.lower().replace("_", "")))

