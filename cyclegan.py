import os

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from torch.utils.data import sampler
from torchvision import datasets, transforms
from itertools import chain

from models import *

dtype = torch.FloatTensor
IMG_DIM = 28

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class CycleGan():

    def __init__(self, generatorAB, generatorBA, discriminatorA, distriminatorB):
        self.G_AB = generatorAB
        self.G_BA = generatorBA
        self.D_A = discriminatorA
        self.D_B = distriminatorB
        self.img_size = 28
        self.batch_size = 128
        self.num_epochs = 10
        self.every = 50
        self.d_optim = torch.optim.Adam(chain(self.D_A.parameters(), self.D_B.parameters()), lr=0.0002, betas=(0.5,
                                                                                                              1 - 1e-3))
        self.g_optim = torch.optim.Adam(chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=0.0002, betas=(0.5,
                                                                                                            1 - 1e-3))
        # self.gb_optim = torch.optim.Adam(, lr=0.0002, betas=(0.5, 1 - 1e-3))
        self.realy = torch.ones(self.batch_size, 1)
        self.fakey = torch.zeros(self.batch_size, 1)
        self.id_loss = nn.L1Loss()
        self.adv_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.load_data()


    def load_data(self):
        transform = transforms.Compose(
            [
                # transforms.Resize(20),
                transforms.ToTensor(),
            ])
        dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset1, batch_size=self.batch_size)


    def train_D(self, D, G, xa, xb, y):
        D.zero_grad()
        with torch.no_grad():
            xba = G(xb)
        real_scores = D(xa, y)
        fake_scores = D(xba, y)
        real_loss = self.adv_loss(real_scores, self.realy)
        fake_loss = self.adv_loss(fake_scores, self.fakey)
        loss = real_loss + fake_loss
        loss.backward()
        return real_loss, fake_loss


    def train(self, name):
        if not hasattr('self', 'train_loader'):
            print("loading data")
            self.load_data()
        print("Starting training")
        counter = 0
        for epoch in range(self.num_epochs):
            for xa, y in self.train_loader:
                if len(xa) != self.batch_size:
                    continue

                #invert colors and flip 90 degrees
                xb = 1 - torch.rot90(xa, 1, [-1,-2])

                xa = 2 * (xa - 0.5)
                xb = 2 * (xb - 0.5)

                real_loss_DA, fake_loss_DA = self.train_D(self.D_A, self.G_BA, xa, xb, y)
                real_loss_DB, fake_loss_DB = self.train_D(self.D_B, self.G_AB, xb, xa, y)
                self.d_optim.step()

                self.G_AB.zero_grad()
                self.G_BA.zero_grad()

                xab = self.G_AB(xa)
                xaba = self.G_BA(xab)
                g_ab_adv_loss = self.adv_loss(self.D_B(xab, y), self.realy)
                g_ab_id_loss = self.id_loss(self.G_AB(xb), xb)
                g_ab_cycle_loss = self.cycle_loss(xaba, xa)
                g_ab_loss = 1 * g_ab_adv_loss + 5 * g_ab_id_loss + 10 * g_ab_cycle_loss

                xba = self.G_BA(xb)
                xbab = self.G_AB(xba)
                g_ba_adv_loss = self.adv_loss(self.D_A(xba, y), self.realy)
                g_ba_id_loss = self.id_loss(self.G_BA(xa), xa)
                g_ba_cycle_loss = self.cycle_loss(xbab, xb)
                g_ba_loss = 1 * g_ba_adv_loss + 5 * g_ba_id_loss + 10 * g_ba_cycle_loss

                g_ab_loss.backward()
                g_ba_loss.backward()
                self.g_optim.step()
                # self.gb_optim.step()

                if counter % self.every == self.every - 1:
                    self.log_status(counter, real_loss_DA, fake_loss_DA, real_loss_DB, fake_loss_DB, g_ab_adv_loss,
                                    g_ab_id_loss, g_ab_cycle_loss, g_ba_adv_loss,
                                    g_ba_id_loss, g_ba_cycle_loss,
                                    xa, xb, xab, xba, xaba, xbab, name)
                counter += 1


    def log_status(self, counter, real_loss_DA, fake_loss_DA, real_loss_DB, fake_loss_DB, g_ab_adv_loss,
                   g_ab_id_loss, g_ab_cycle_loss, g_ba_adv_loss,
                   g_ba_id_loss, g_ba_cycle_loss,
                   xa, xb, xab, xba, xaba, xbab, name):
        print(f"\n\nStep {counter}")
        print(f"real_loss_DA: {real_loss_DA}  fake_loss_DA: {fake_loss_DA}")
        print(f"real_loss_DB: {real_loss_DB}  fake_loss_DB: {fake_loss_DB}")
        print(f"g_ab_adv_loss: {g_ab_adv_loss}  g_ab_id_loss: {g_ab_id_loss}  g_ab_cycle_loss: {g_ab_cycle_loss}")
        print(f"g_ba_adv_loss: {g_ba_adv_loss}  g_ba_id_loss: {g_ba_id_loss}  g_ba_cycle_loss: {g_ba_cycle_loss}")

        xa = xa[:5].numpy().reshape(-1, self.img_size, self.img_size)
        xb = xb[:5].numpy().reshape(-1, self.img_size, self.img_size)
        xab = xab[:5].detach().numpy().reshape(-1, self.img_size, self.img_size)
        xba = xba[:5].detach().numpy().reshape(-1, self.img_size, self.img_size)
        xaba = xaba[:5].detach().numpy().reshape(-1, self.img_size, self.img_size)
        xbab = xbab[:5].detach().numpy().reshape(-1, self.img_size, self.img_size)
        real = np.concatenate([xa, xb], axis=0)
        fake1 = np.concatenate([xab, xba], axis=0)
        fake2 = np.concatenate([xaba, xbab], axis=0)
        self.show_images(real, fake1, fake2, f"{name}_imgs_step_{counter}.png")
        torch.save(self.D_A, f"models/discriminatorA_{counter}")
        torch.save(self.D_B, f"models/discriminatorB_{counter}")
        torch.save(self.G_AB, f"models/generatorAB_{counter}")
        torch.save(self.G_BA, f"models/generatorBA_{counter}")


    def show_images(self, real_images, fake_images1, fake_images2, filename=None):
        # images = images[:max_imgs].view(-1,28,28)
        gridlen = len(real_images)
        fig = plt.figure(figsize=(gridlen, 3))
        gs = gridspec.GridSpec(3, gridlen)
        gs.update(wspace=0.05, hspace=0.05)
        images = np.concatenate([real_images, fake_images1, fake_images2], axis=0)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img)
            if i == gridlen ** 2 - 1:
                break
        if filename:
            plt.savefig(os.path.join("fake_imgs", filename))
        else:
            plt.show()


    def sample(self):
        G_AB = torch.load("generatorAB_499")
        G_BA = torch.load("generatorBA_499")
        G_AB.eval()
        G_BA.eval()
        with torch.no_grad():
            for x, _ in self.train_loader:
                xa = x[:5]
                xb = 1 - xa
                xa = 2 * (xa - 0.5)
                xb = 2 * (xb - 0.5)
                fakexab = G_AB(xa).numpy()
                fakexba = G_BA(xb).numpy()
                fake = np.concatenate([fakexab, fakexba], axis=0).squeeze()
                real = np.concatenate([xa, xb], axis=0).squeeze()
                self.show_images(real, fake)


if __name__ == "__main__":
    d_params = {"img_dim": IMG_DIM,
                "num_conv1": 32,
                "size_conv1": 5,
                "num_conv2": 64,
                "size_conv2": 5,
                "label_emb_dim": 50,
                "n_classes": 10
                }

    G1 = CycleGanGenerator()
    G2 = CycleGanGenerator()
    D1 = cDCDiscriminator(**d_params)
    D2 = cDCDiscriminator(**d_params)

    gan = CycleGan(G1, G2, D1, D2)
    gan.train("CycleGAN")
    # gan.sample()
