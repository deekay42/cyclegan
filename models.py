from torch import nn
import torch
import torch.nn.functional as F


class VanillaDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.01)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        scores = self.fc3(x)
        return scores

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim



class VanillaGenerator(Generator):
    def __init__(self, noise_dim, hidden_dim, img_dim):
        super().__init__(noise_dim)
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, img_dim)


    def forward(self, batch_size):
        z = self.noise(batch_size)
        x = self.fc1(z)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x

class DCDiscriminator(nn.Module):

    def __init__(self, img_dim, num_conv1, size_conv1, num_conv2, size_conv2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_conv1, size_conv1, 1)
        self.conv2 = nn.Conv2d(num_conv1, num_conv2, size_conv2, 1)

        self.conv2_out_w = (1 + ((1 + (img_dim - size_conv1)) // 2 - size_conv2)) // 2
        self.fc1_in = self.conv2_out_w * self.conv2_out_w * num_conv2

        self.fc1 = nn.Linear(self.fc1_in, self.fc1_in)
        self.fc2 = nn.Linear(self.fc1_in, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = torch.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.max_pool2d(x, 2)

        x = x.view(-1, self.fc1_in)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)

        return x


class DCGenerator(Generator):

    def __init__(self, noise_dim, hidden_dim, num_conv1, size_conv1, num_conv2, size_conv2, img_dim):
        super().__init__(noise_dim)
        self.conv2_in_w = 1 + (img_dim - size_conv2 + 2) // 2
        self.conv1_in_w = 1 + (self.conv2_in_w - size_conv1 + 2) // 2
        self.fc2_in = self.conv1_in_w * self.conv1_in_w * num_conv1
        self.num_conv1 = num_conv1

        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, self.fc2_in)
        self.bn2 = nn.BatchNorm1d(self.fc2_in)

        self.deconv1 = nn.ConvTranspose2d(num_conv1, num_conv2, size_conv1, 2, 1)
        self.bn3 = nn.BatchNorm2d(num_conv2)
        self.deconv2 = nn.ConvTranspose2d(num_conv2, 1, size_conv2, 2, 1)


    def forward(self, z):

        x = self.fc1(z)
        x = torch.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)

        x = x.view(-1, self.num_conv1, self.conv1_in_w, self.conv1_in_w)
        x = self.deconv1(x)
        x = torch.relu(x)

        x = self.bn3(x)
        x = self.deconv2(x)
        x = torch.tanh(x)

        return x


class cDCDiscriminator(DCDiscriminator):
    def __init__(self, img_dim, num_conv1, size_conv1, num_conv2, size_conv2, label_emb_dim, n_classes):
        super().__init__(img_dim, num_conv1, size_conv1, num_conv2, size_conv2)
        self.label_emb_dim = label_emb_dim
        self.n_classes = n_classes
        self.emb = nn.Embedding(self.n_classes, self.label_emb_dim)
        self.label_fc = nn.Linear(self.label_emb_dim, img_dim**2)
        self.conv1 = nn.Conv2d(1+1, num_conv1, size_conv1, 1)

    def forward(self, x, label):
        label_emb = self.emb(label)
        label_fc = self.label_fc(label_emb)
        label_fc = label_fc.view(-1, 1, *x.shape[2:])
        x = torch.cat((x, label_fc), dim=1)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        # x = self.bn1(x)
        x = torch.max_pool2d(x, 2)

        # x = torch.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        # x = self.bn2(x)
        x = torch.max_pool2d(x, 2)

        # x = torch.max_pool2d(x, 2)

        x = x.view(-1, self.fc1_in)
        x = self.fc2(x)

        return x



class cDCGenerator(DCGenerator):
    def __init__(self, noise_dim, num_conv1, size_conv1, num_conv2, size_conv2, img_dim, label_emb_dim, n_classes):
        super().__init__(noise_dim, num_conv1, size_conv1, num_conv2, size_conv2, img_dim)
        self.label_emb_dim = label_emb_dim
        self.n_classes = n_classes
        self.emb = nn.Embedding(self.n_classes, self.label_emb_dim)
        self.label_fc = nn.Linear(self.label_emb_dim, self.conv1_in_w * self.conv1_in_w)
        self.deconv1 = nn.ConvTranspose2d(num_conv1+1, num_conv2, size_conv1, 2, 1)

    def forward(self, z, label):
        label_emb = self.emb(label)
        label_fc = self.label_fc(label_emb)
        label_fc = label_fc.view(-1, 1, self.conv1_in_w, self.conv1_in_w)

        x = self.fc1(z)
        x = torch.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)

        x = x.view(-1, self.num_conv1, self.conv1_in_w, self.conv1_in_w)
        x = torch.cat((x, label_fc), dim=1)
        x = self.deconv1(x)
        x = torch.relu(x)
        x = self.bn3(x)

        x = self.deconv2(x)
        x = torch.tanh(x)

        return x


class CycleGanGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(64 * 4 * 4, 4 * 4 * 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 128),
            Unflatten(-1, 128, 7, 7),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )


    def forward(self, img):
        return self.model(img)


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)
