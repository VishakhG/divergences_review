import matplotlib
matplotlib.use('Agg')
from args import get_args # noqa
import torch.nn as nn # noqa
import torch.nn.functional as F # noqa
from torch.utils.data  import DataLoader # noqa
import torch # noqa
from torch.autograd import Variable # noqa
from gmm import GMM # noqa
import matplotlib.pyplot as plt # noqa
from itertools import islice # noqa
import seaborn as sns # noqa
import numpy as np # noqa
from utils import pdf_normal, plot_density_data # noqa
from utils import Generator # noqa


class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        output = F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x))))
        return output


def energy(X):
    recon = netD(X)
    return torch.mean(torch.sum((X - recon)**2, 1))


args = get_args()

n_samples = args.n_data
n_epochs = args.n_epochs
batch_size = args.batch_size
m = 5
gmm = GMM(n_gaussians=2, dim=2, random_seed=22)
sample_data = torch.Tensor(gmm.sample(n_samples))

dataloader_train = DataLoader(sample_data, batch_size=batch_size)


noise = torch.rand(n_samples, 2)
fixed_noise = torch.rand(n_samples, 2)

netG = Generator(sample_data.numpy())
netD = Discrimator()

if torch.cuda.is_available():
    netG = netG.cuda()
    netD = netD.cuda()
    fixed_noise = fixed_noise.cuda()

optim_netD = torch.optim.Adam(netD.parameters(), lr=args.lr)
optim_netG = torch.optim.Adam(netG.parameters(), lr=args.lr)

for epoch_i in range(n_epochs):
    epoch_discriminator_losses = []
    epoch_generator_losses = []

    for _, batch_real in enumerate(dataloader_train):
        batch_real = Variable(batch_real)
        z = Variable(torch.rand(batch_size, 2, 2))
        if torch.cuda.is_available():
            batch_real = batch_real.cuda()
            z = z.cuda()

        batch_fake = netG(z)

        optim_netD.zero_grad()
        d_real = energy(batch_real)
        d_fake = energy(batch_fake)
        discriminator_loss = d_real + F.leaky_relu(m - d_fake)
        epoch_discriminator_losses.append(float(discriminator_loss.data[0]))
        discriminator_loss.backward()
        optim_netD.step()

        optim_netG.zero_grad()
        batch_fake = netG(z)
        d_fake = energy(batch_fake)
        epoch_generator_losses.append(float(d_fake.data[0]))
        d_fake.backward()
        optim_netG.step()

    if epoch_i % args.save_freq == 0:
        print("Batch {0} generator loss {1} discriminator_loss {2}".format(
            epoch_i, np.mean(epoch_discriminator_losses),
            np.mean(epoch_generator_losses)))

if torch.cuda.is_available():
    gen_data = netG(Variable(fixed_noise, volatile=True)).data.cpu().numpy()
else:
    with torch.no_grad():
        gen_data = netG(Variable(fixed_noise)).data.numpy()

plt.title("Energy based gans")

sample_data = sample_data.cpu().numpy()
mu, std = netG.get_params()
mu = mu.cpu().data.numpy()
std = std.cpu().data.numpy()
std = np.fill_diagonal(np.zeros((mu.shape[0], mu.shape[0])), std[0])
pdf = pdf_normal(mu, std)
plot_density_data(pdf, sample_data)
plt.savefig("../plots/eb_gan_samples.png")
