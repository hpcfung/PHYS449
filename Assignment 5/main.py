import numpy as np
import torch, json, os, argparse
from torch import nn
import matplotlib.pyplot as plt

# dataset for training
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        # the first 3000 images are reserved for calculating the test loss
        self.train_data = raw_data[3000:,0:196]

    def __len__(self):
        return self.train_data.size()[0]

    def __getitem__(self, idx):
        return self.train_data[idx,:]

# dataset for calculating the test loss
class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        # the first 3000 images are reserved for calculating the test loss
        self.test_data = raw_data[0:3000,0:196]

    def __len__(self):
        return self.test_data.size()[0]

    def __getitem__(self, idx):
        return self.test_data[idx,:]

# the encoder part of the VAE
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layer1_kernels = 3
        self.conv_layer2_kernels = 3

        self.fc_layer1_width = 14*14*self.conv_layer2_kernels

        self.encode_layers = nn.Sequential(
            nn.Conv2d(1,self.conv_layer1_kernels,3,padding=(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(self.conv_layer1_kernels, self.conv_layer2_kernels, 3, padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(self.fc_layer1_width, 2*latent_dim),
        )

    def forward(self, x1):
        raw_model_output = self.encode_layers(x1)
        _MEAN = raw_model_output[:,0:latent_dim]
        # +1e-20 makes sure the standard deviation is nonzero so that the Gaussian is well-defined
        _SD = nn.functional.relu(raw_model_output[:,latent_dim:latent_dim*2])+1e-20
        return _MEAN, _SD

# the decoder part of the VAE
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

# implements the closed-form formula for the KL-divergence, which is the regularizer
# see README.md for details
def regularizer(_mean,_sd):
    mean_sq = _mean ** 2
    sd_sq = _sd ** 2
    pre_sum = -torch.log(sd_sq)+mean_sq+sd_sq
    # dim = 1 indexes the latent space; we sum over this dimension
    pre_avg = torch.sum(pre_sum,1)
    # this computes the average over the whole batch
    return torch.mean(pre_avg)

# training the VAE
def training():
    # the train loss and test loss are stored in these lists, which are used to generate the loss plot
    train_loss_history = []
    test_loss_history = []
    for epoch in range(num_epoch):
        # stores the training loss in each batch
        train_loss_per_batch = []
        for batch, img_in_seq in enumerate(train_dataloader):
            # In a VAE, we assume the covariance is diagonal
            # which is equivalent to sampling from independent Gaussians, eg see
            # https://stats.stackexchange.com/questions/198362/how-to-define-a-2d-gaussian-using-1d-variance-of-component-gaussians

            # sampling from the standard Gaussian as per the re-parametrization trick
            epsilon = torch.normal(0, 1, size=(len(img_in_seq) * L, latent_dim)).to(device)

            img_batch = pre_processing(img_in_seq)

            # the encoder outputs the mean and standard deviation
            mean, sd = encoder_model(img_batch)

            # we sample each latent vector L times to obtain a better estimate of the expectation value
            # hence we repeat these tensors
            repeated_mean = mean.repeat(L, 1)
            repeated_sd = sd.repeat(L, 1)

            # the latent vector is sampled using the re-parametrization trick
            latent_vec = repeated_mean + repeated_sd * epsilon

            # the decoder outputs the reconstructed image
            recon_img_batch = decoder_model(latent_vec)

            # repeat the original images L times so that it is of the same shape as the reconstructed images
            repeated_img_batch = img_in_seq.repeat(L, 1)

            # reconstruction loss; see README.md for details
            recon_loss = nn.functional.mse_loss(recon_img_batch, repeated_img_batch)
            # total loss
            loss = recon_loss + _lambda * regularizer(mean, sd)

            train_loss_per_batch.append(loss.item())

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaging the the training loss over the entire epoch
        train_loss = sum(train_loss_per_batch)/len(train_loss_per_batch)
        train_loss_history.append(train_loss)

        # computes the test loss
        # the code is the same as that for the training loss
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
            test_loss_history.append(test_loss.item())

        # in the verbose mode, gives iterative reports on the loss value after every epoch
        if verbosity == 2:
            print(f"epoch = {epoch + 1}   training loss = {train_loss}   testing loss = {test_loss}")
        else:
            if (epoch+1)%10==0:
                print(f"{epoch + 1} epochs completed")

    print(f"Training complete\n")

    # generates the loss plot
    plt.plot(np.arange(1,num_epoch+1),train_loss_history,label='training loss')
    plt.plot(np.arange(1,num_epoch+1),test_loss_history,label='testing loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training progress of the VAE')
    plt.savefig(results_path + '/loss.pdf')
    plt.clf()

# testing the VAE: generates digits by sampling latent vectors from a Gaussian
def testing():
    print(f"Generating plots")
    latent_vec = torch.normal(0,3,size=(num_plots,latent_dim)).to(device)
    generated_img_batch = decoder_model(latent_vec)
    plot_imgs(generated_img_batch)

# plots the sampled images
def plot_imgs(batch_pics):
    # for each image in the batch of outputs
    for _I in range(len(batch_pics)):
        # the decoder output is a 1D array
        # this reshapes the output so that we can plot it as a 2D picture
        picture = torch.reshape(batch_pics[_I], (14, 14)).detach().cpu()
        plt.axis("off")
        plt.imshow(picture, cmap="gray")
        plt.savefig(results_path +'/'+str(_I+1)+ '.pdf')
        if (_I+1)%10==0:
            print(f"{_I+1} digits sampled")

if __name__ == '__main__':
    device = torch.device("cpu")

    # Command line arguments
    parser = argparse.ArgumentParser(description='Using a variational auto-encoder (VAE) to learn '
                                                 'to write even numbers')
    parser.add_argument('-o', type=str, default='result_dir',
                        help='folder where the result files are stored in')
    parser.add_argument('-n', type=int, default=100, help='number of sampled digit images')
    
    args = parser.parse_args()

    # generates the folder in which the results will be stored
    results_path = args.o
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # number of sampled digits
    num_plots = args.n

    # read param.json file
    g = open('param.json')
    data = json.load(g)

    # loads the .csv file as a tensor
    raw_data = torch.tensor(np.loadtxt('data/even_mnist.csv', dtype=np.float32, usecols=range(0, 196))).to(device)

    # loading the parameters from param.json, see README.md for details
    Batch_size = data['Batch size']
    latent_dim = data['Latent dimension']
    L = data['L'] # sample the same input image L times to get a better estimate of the expectation value

    learning_rate = data['Learning rate']
    _lambda = data['Regularizer']
    num_epoch = data['Number of epochs']

    verbosity = data['verbosity']

    print(f"Loading complete\n")

    # initialization of the dataloader, neural networks, and the optimizer
    train_dataset = TrainDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_dataset = TestDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=False)

    encoder_model = Encoder().to(device)
    decoder_model = Decoder().to(device)

    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=learning_rate)

    # training and testing
    training()
    testing()
    print("\nAll plots generated, program is complete")
