import numpy as np
import torch, sys, json, os, argparse
from torch import nn
import matplotlib.pyplot as plt

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.train_data = raw_data[3000:,0:196]

    def __len__(self):
        return self.train_data.size()[0]

    def __getitem__(self, idx):
        return self.train_data[idx,:]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.test_data = raw_data[0:3000,0:196]

    def __len__(self):
        return self.test_data.size()[0]

    def __getitem__(self, idx):
        return self.test_data[idx,:]

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

        self.decode_layers = nn.Sequential(
            nn.Linear(latent_dim, self.decode_layer1_width),
            nn.LeakyReLU(),
            nn.Linear(self.decode_layer1_width, self.decode_layer2_width),
            nn.LeakyReLU(),
            nn.Linear(self.decode_layer2_width, 14 * 14),
        )

    def forward(self, x2):
        return self.decode_layers(x2)

# the output of the dataloader is a 1D array for each image
# this converts it into a 2D array, so that it can be fed into the CNN
def pre_processing(seqs):
    current_batch_size = len(seqs)
    _2D_img = torch.reshape(seqs, (current_batch_size,14,14))
    # add one more dimension, since number of channels = 1
    # CNN input is of the form (batch size, channels, height, width)
    return torch.unsqueeze(_2D_img,1)

# https://stats.stackexchange.com/questions/198362/how-to-define-a-2d-gaussian-using-1d-variance-of-component-gaussians
# diagonal covariance = independent Gaussian

def regularizer(_mean,_sd):
    mean_sq = _mean ** 2
    sd_sq = _sd ** 2
    pre_sum = -torch.log(sd_sq)+mean_sq+sd_sq
    pre_avg = torch.sum(pre_sum,1)
    return torch.mean(pre_avg)


def training():
    train_lost_history = []
    for epoch in range(num_epoch):
        train_loss_per_batch = []
        for batch, img_in_seq in enumerate(train_dataloader):
            # sample from standard Gaussian as per the re-parametrization trick
            epsilon = torch.normal(0, 1, size=(len(img_in_seq) * L, latent_dim)).to(device)

            img_batch = pre_processing(img_in_seq)
            mean, sd = encoder_model(img_batch)

            repeated_mean = mean.repeat(L, 1)
            repeated_sd = sd.repeat(L, 1)

            latent_vec = repeated_mean + repeated_sd * epsilon

            recon_img_batch = decoder_model(latent_vec)

            repeated_img_batch = img_in_seq.repeat(L, 1)

            recon_loss = nn.functional.mse_loss(recon_img_batch, repeated_img_batch)
            loss = recon_loss + _lambda * regularizer(mean, sd)

            train_loss_per_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = sum(train_loss_per_batch)/len(train_loss_per_batch)
        train_lost_history.append(train_loss)
        if calc_test_loss:
            for batch2, img_in_seq2 in enumerate(test_dataloader):
                epsilon2 = torch.normal(0, 1, size=(len(img_in_seq2) * L, latent_dim)).to(device)

                img_batch2 = pre_processing(img_in_seq2)
                mean2, sd2 = encoder_model(img_batch2)

                repeated_mean2 = mean2.repeat(L, 1)
                repeated_sd2 = sd2.repeat(L, 1)

                latent_vec2 = repeated_mean2 + repeated_sd2 * epsilon2

                recon_img_batch2 = decoder_model(latent_vec2)

                repeated_img_batch2 = img_in_seq2.repeat(L, 1)

                recon_loss2 = nn.functional.mse_loss(recon_img_batch2, repeated_img_batch2)
                test_loss = recon_loss2 + _lambda * regularizer(mean2, sd2)

                print(f"epoch = {epoch + 1}   training loss = {train_loss}   testing loss = {test_loss}")
        else:
            print(f"epoch = {epoch + 1}   training loss = {train_loss}")

    print(f"\nTraining complete")
    plt.plot(np.arange(1,num_epoch+1),train_lost_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(results_path + '/loss.pdf')

def testing():
    print(f"Generating plots")
    latent_vec = torch.normal(0,3,size=(num_plots,latent_dim)).to(device)
    generated_img_batch = decoder_model(latent_vec)
    tmp_plot(generated_img_batch)

def tmp_plot(batch_pics):
    for _I in range(len(batch_pics)):
        # plt.figure()
        picture = torch.reshape(batch_pics[_I], (14, 14)).detach().cpu()
        plt.axis("off")
        plt.imshow(picture, cmap="gray")
        plt.savefig(results_path +'/'+str(_I+1)+ '.pdf')
        # plt.clf()
        if (_I+1)%10==0:
            print(f"{_I+1} digits sampled")

if __name__ == '__main__':
    # set this to cpu at the end
    device = torch.device("cpu")

    # Command line arguments
    parser = argparse.ArgumentParser(description='Using a variational auto-encoder (VAE) to learn '
                                                 'to write even numbers')
    parser.add_argument('-o', type=str, default='result_dir',
                        help='folder where the result files are stored in')
    parser.add_argument('-n', type=int, default=100, help='number of sampled digit images')
    # parser.add_argument('-n', type=int, default=100, help='verbosity (default: 1)')

    args = parser.parse_args()

    # print(args.o)
    # print(args.n)
    # sys.exit()

    # generates the folder in which the results will be stored
    results_path = args.o # 'result_dir'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    num_plots = 5

    # read json file
    g = open('param.json')
    data = json.load(g)

    raw_data = torch.tensor(np.loadtxt('data/even_mnist.csv', dtype=np.float32, usecols=range(0, 196))).to(device)

    Batch_size = data['Batch size']
    latent_dim = data['Latent dimension']
    L = data['L'] # sample the same image L times to get a better estimate of the expectation value

    learning_rate = data['Learning rate']
    _lambda = data['Regularizer']
    num_epoch = data['Number of epochs']

    calc_test_loss = False

    train_dataset = TrainDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_dataset = TestDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=False)

    encoder_model = Encoder().to(device)
    decoder_model = Decoder().to(device)

    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=learning_rate)

    training()
    testing()
    print("All plots generated, program is complete")
