import matplotlib
matplotlib.use('Agg')
import torch.nn as nn # noqa
import torch.nn.functional as F # noqa
from torch.utils.data  import DataLoader # noqa
import torch # noqa
from torch.autograd import Variable # noqa
from gmm import GMM # noqa
import matplotlib.pyplot as plt # noqa
import random # noqa 
from args import get_args # noqa
import numpy as np # noqa
from utils import pdf_normal # noqa
from utils import Generator, plot_density_data # noqa


args = get_args()

manualSeed = args.random_seed  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# BX2


class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        output = F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x))))
        output = output.mean(0)
        return output.view(1)


gmm = GMM(n_gaussians=2, dim=2, random_seed=22)

discrimator_iters = 100
generator_iters = 25
n_samples = args.n_data
n_epochs = args.n_epochs
batch_size = args.batch_size

sample_data = torch.Tensor(gmm.sample(n_samples))
dataloader_train = DataLoader(sample_data, batch_size=batch_size, shuffle=True)
one = torch.FloatTensor([1])
neg_one = one * -1
noise = torch.rand(n_samples, 2)
fixed_noise = torch.rand(n_samples, 2)

netG = Generator(sample_data.numpy())
netD = Discrimator()

optim_netD = torch.optim.Adam(netD.parameters(), lr=args.lr)
optim_netG = torch.optim.Adam(netG.parameters(), lr=args.lr)


if torch.cuda.is_available():
    netG = netG.cuda()
    netD = netD.cuda()
    sample_data = sample_data.cuda()
    one = one.cuda()
    neg_one = neg_one.cuda()
    fixed_noise = fixed_noise.cuda()

for epoch_i in range(n_epochs):
    if epoch_i % 10 == 0:
        print(epoch_i)

    dataloader_iter = iter(dataloader_train)
    total_counter = 0
    epoch_generator_losses = []
    epoch_discriminator_losses = []
    while total_counter < len(dataloader_iter):
        netD_counter = 0
        netG_counter = 0

        while netD_counter < discrimator_iters:
            batch_real = dataloader_iter.next()
            if torch.cuda.is_available():
                batch_real = batch_real.cuda()
            total_counter += 1

            # Lipshitz constraint
            for params in netD.parameters():
                    params.data.clamp_(-.01, .01)

            netD.zero_grad()
            d_error_real = netD(Variable(batch_real))
            d_error_real.backward(one)

            noise = torch.rand(batch_size, 2)
            if torch.cuda.is_available():
                noise = noise.cuda()

            if torch.cuda.is_available():
                batch_fake = Variable(
                    netG(Variable(noise, volatile=True)).data)
            else:
                with torch.no_grad():
                    batch_fake = Variable(netG(Variable(noise)).data)

            d_error_fake = netD(batch_fake)
            d_error_fake.backward(neg_one)

            d_error_total = d_error_real - d_error_fake
            epoch_discriminator_losses.append(d_error_total.data[0])
          
            optim_netD.step()
            netD_counter += 1

        while netG_counter < generator_iters:
            netG.zero_grad()
            noise = torch.rand(batch_size, 2)
            if torch.cuda.is_available():
                noise = noise.cuda()
            fake = netG(Variable(noise))
            g_error = netD(fake)
            epoch_generator_losses.append(g_error.data[0])
            g_error.backward(one)

            optim_netG.step()
            netG_counter += 1

    if epoch_i % args.save_freq == 0:
        print("Batch {0} generator loss {1} discriminator_loss {2}".format(
            epoch_i, np.mean(epoch_discriminator_losses), np.mean(
                epoch_generator_losses)))

if torch.cuda.is_available():
    gen_data = netG(Variable(fixed_noise, volatile=True)).data.cpu().numpy()
else:
    with torch.no_grad():
        gen_data = netG(Variable(fixed_noise)).data.numpy()

sample_data = sample_data.cpu().numpy()
mu, std = netG.get_params()
mu = mu.cpu().data.numpy()
std = std.cpu().data.numpy()
std = np.fill_diagonal(np.zeros((mu.shape[0], mu.shape[0])), std[0])
pdf = pdf_normal(mu, std)
plot_density_data(pdf, sample_data)
plt.savefig("../plots/wasserstein_gan_samples.png")
