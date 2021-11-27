import numpy as np
import torch, random, time, sys
from torch import nn
import matplotlib.pyplot as plt

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.raw_data = torch.tensor(np.loadtxt(data_path,dtype=np.float32,usecols=range(0, 196))).to(device)

    def __len__(self):
        return self.raw_data.size()[0]

    def __getitem__(self, idx):
        return self.raw_data[idx,:]

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layer1_kernels = 3
        self.conv_layer2_kernels = 3
        # self.conv_layer3_kernels = 3

        self.fc_layer1_width = 14*14*self.conv_layer2_kernels
        # self.fc_layer2_width = 50

        self.encode_layers = nn.Sequential(
            nn.Conv2d(1,self.conv_layer1_kernels,3,padding=(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(self.conv_layer1_kernels, self.conv_layer2_kernels, 3, padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),

            # nn.Linear(self.fc_layer1_width, self.fc_layer2_width),
            # nn.LeakyReLU(),
            nn.Linear(self.fc_layer1_width, 2*latent_dim),
        )

    def forward(self, x1):
        raw_model_output = self.encode_layers(x1)
        _MEAN = raw_model_output[:,0:latent_dim]
        _SD = nn.functional.relu(raw_model_output[:,latent_dim:latent_dim*2])+1e-20
        # make sure the standard deviation is nonzero so that the Gaussian is well-defined
        return _MEAN, _SD

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode_layer1_width = 100
        self.decode_layer2_width = 196
        # self.decode_layer3_width = 150

        self.decode_layers = nn.Sequential(
            nn.Linear(latent_dim, self.decode_layer1_width),
            nn.LeakyReLU(),
            nn.Linear(self.decode_layer1_width, self.decode_layer2_width),
            nn.LeakyReLU(),
            nn.Linear(self.decode_layer2_width, 14 * 14),
        )

    def forward(self, x2):
        # print(self.conv_layers(x).size())
        # sys.exit()
        return self.decode_layers(x2)

def pre_processing(seqs):
    current_batch_size = len(seqs)
    _2D_img = torch.reshape(seqs, (current_batch_size,14,14))
    return torch.unsqueeze(_2D_img,1)

def tmp_plot(batch_pics):
    fig = plt.figure()
    for _I in range(len(batch_pics)):
        picture = torch.reshape(batch_pics[_I], (14, 14)).detach().cpu()
        fig.add_subplot(10, 20, _I+1)
        plt.axis("off")
        plt.imshow(picture, cmap="gray")  # np.array(picture)

# input of shape (batch size, channel = 1, 14, 14)
def raw_plot(batch_pics):
    fig = plt.figure()
    for _I in range(len(batch_pics)):
        picture = torch.squeeze(batch_pics[_I], dim=0).detach().cpu()
        fig.add_subplot(10, 10, _I+1)
        plt.axis("off")
        plt.imshow(picture, cmap="gray")  # np.array(picture)

# https://stats.stackexchange.com/questions/198362/how-to-define-a-2d-gaussian-using-1d-variance-of-component-gaussians
# diagonal covariance = independent Gaussian

def regularizer(_mean,_sd):
    mean_sq = _mean ** 2
    sd_sq = _sd ** 2
    pre_sum = -torch.log(sd_sq)+mean_sq+sd_sq
    pre_avg = torch.sum(pre_sum,1)
    return torch.mean(pre_avg)


def training():
    for epoch in range(num_epoch):
        print(f'epoch {epoch + 1}')
        for batch, img_in_seq in enumerate(train_dataloader):
            # sample from standard Gaussian as per the re-parametrization trick
            epsilon = torch.normal(0, 1, size=(len(img_in_seq) * L, latent_dim))
            # print(epsilon.size())

            # tmp_plot(img)
            # plt.show()
            # print(img.size())
            # raw_plot(img)
            # plt.show()

            # print(model(img_batch).size())
            img_batch = pre_processing(img_in_seq)
            # print(img_batch.size())

            mean, sd = encoder_model(img_batch)
            # print(mean.size())
            # print(sd.size())
            #
            # print(mean)
            # print(sd)
            # print(epsilon)

            repeated_mean = mean.repeat(L, 1)
            repeated_sd = sd.repeat(L, 1)

            # print(repeated_mean.size())
            # print(repeated_sd.size())

            latent_vec = repeated_mean + repeated_sd * epsilon
            # print(latent_vec.size())
            # print(latent_vec)

            recon_img_batch = decoder_model(latent_vec)
            # print(recon_img_batch.size())

            # print(img_batch.size())
            repeated_img_batch = img_in_seq.repeat(L, 1)

            recon_loss = nn.functional.mse_loss(recon_img_batch, repeated_img_batch)
            loss = recon_loss + _lambda * regularizer(mean, sd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"batch = {batch}   loss = {loss}")
                # print(img.size())
            if batch == 400:
                print(recon_img_batch.size())
                tmp_plot(recon_img_batch)
                plt.show()
                sys.exit()


if __name__ == '__main__':
    device = torch.device("cpu")

    data_path = 'data/even_mnist.csv'
    Batch_size = 64
    latent_dim = 4
    L = 3 # sample the same image L times to get a better estimate of the expectation value

    learning_rate = 1e-3
    _lambda = 1e-1
    num_epoch = 2

    train_dataset = MNISTDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

    encoder_model = Encoder().to(device)
    decoder_model = Decoder().to(device)

    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=learning_rate)

    training()



    # latent_vec = torch.normal(mean,sd)
    # nope, can't do backprop with this
    # since no computational graph from encoder weights

    # print(latent_vec.size())



