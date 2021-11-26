# PHYS449

## Dependencies

- json
- numpy

## Running `main.py`

To run `main.py`, use

```sh
python main.py
```

## Derivation of the Loss Function

As shown in Lecture15.pdf, after the re-parametrization trick, the evidence lower bound becomes

<img src="https://render.githubusercontent.com/render/math?math=\log{p(x)}\geq\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]-KL(q_\theta(z|x)\|p(z))">

Since we want to maximize the likelihood <img src="https://render.githubusercontent.com/render/math?math=p(x)">, we maximize the lower bound <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]-KL(q_\theta(z|x)\|p(z))">. Hence the loss function, the function we minimize during training, is <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}=-\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]%2BKL(q_\theta(z|x)\|p(z))">. These two terms can be further simplified.

### Reconstruction Likelihood

The reconstruction likelihood is <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]">. As stated in Section 3 of the original paper on variational auto-encoders (VAE)[^1], for real-valued data, a Gaussian multilayer perceptron is used for the decoder. As usual, by assuming a Gaussian noise model, we can derive the mean square error function. The details are as follows.

Using the law of large numbers, we can approximate <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]"> by sampling a sufficently large number of epsilons

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]\simeq\frac{1}{L}\sum_{l=1}^L\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon^{(l)})}">

Consider each <img src="https://render.githubusercontent.com/render/math?math=\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon^{(l)})}">. The output <img src="https://render.githubusercontent.com/render/math?math=x"> is an image with <img src="https://render.githubusercontent.com/render/math?math=N=14\times14"> number of pixels. Let <img src="https://render.githubusercontent.com/render/math?math=x_\phi=x_\phi(z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon^{(l)})"> be the output of the decoder, ie the predicted image, when its input is <img src="https://render.githubusercontent.com/render/math?math=z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon^{(l)}">. Let <img src="https://render.githubusercontent.com/render/math?math=x_{(n)}"> and <img src="https://render.githubusercontent.com/render/math?math=x_{\phi,(n)}"> be the <img src="https://render.githubusercontent.com/render/math?math=n">th pixel in <img src="https://render.githubusercontent.com/render/math?math=x"> and <img src="https://render.githubusercontent.com/render/math?math=x_{\phi}"> respectively. Assume that the Gaussian noise at each pixel is independent of the noise at other pixels. Then the likelihood becomes

<img src="https://render.githubusercontent.com/render/math?math=p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon^{(l)})=\prod_{n=1}^N\mathcal{N}(x_{(n)}|x_{\phi,(n)},\beta^{-1})">

where <img src="https://render.githubusercontent.com/render/math?math=\beta^{-1}"> is the variance, assumed to be some fixed constant. So the log likelihood is

<img src="https://render.githubusercontent.com/render/math?math=\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon^{(l)})}=\sum_{n=1}^N\log{\mathcal{N}(x_{(n)}|x_{\phi,(n)},\beta^{-1})}=\frac{N}{2}\log{\beta}-\frac{N}{2}\log{2\pi}-\frac{\beta}{2}\sum_{n=1}^N(x_{(n)}-x_{\phi,(n)})^2">

<img src="https://render.githubusercontent.com/render/math?math=\frac{N}{2}\log{\beta}-\frac{N}{2}\log{2\pi}"> does not depend on the weights of the decoder <img src="https://render.githubusercontent.com/render/math?math=\phi">, so we can drop it from the loss function.

Note that the loss function can be scaled by any number, and the loss landscape would remain the same (ie the positions of the minima stay the same). Hence we scale <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}"> by <img src="https://render.githubusercontent.com/render/math?math=\frac{2}{\beta N}">, and the reconstruction likelihood term becomes <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{L}\sum_{l=1}^L[\frac{1}{N}\sum_{n=1}^N(x_{(n)}-x_{\phi,(n)})^2]">, which is just the mean squared error.

### Regularizer

Now consider the regularizer <img src="https://render.githubusercontent.com/render/math?math=KL(q_\theta(z|x)\|p(z))">. Let <img src="https://render.githubusercontent.com/render/math?math=J"> be the dimension of the latent space. As stated in Section 3 of the paper[^1], we assume <img src="https://render.githubusercontent.com/render/math?math=q_\theta(z|x)"> is a Gaussian with a diagonal covariance matrix. The output of the encoder is the mean <img src="https://render.githubusercontent.com/render/math?math=\mu_j"> and the diagonal components of the covariance matrix <img src="https://render.githubusercontent.com/render/math?math=\sigma_j"> for <img src="https://render.githubusercontent.com/render/math?math=j=1,\ldots,N">. There is a derivation in Appendix B of the original paper[^1] which allows us to rewrite <img src="https://render.githubusercontent.com/render/math?math=KL(q_\theta(z|x)\|p(z))"> as a simple expression,

<img src="https://render.githubusercontent.com/render/math?math=KL(q_\theta(z|x)\|p(z))=\frac{1}{2}\sum_{j=1}^J(-1-\log{\sigma_{j}^2}%2B\mu_{j}^2%2B\sigma_{j}^2)">

Since we have scaled <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}"> by <img src="https://render.githubusercontent.com/render/math?math=\frac{2}{\beta N}">, this becomes <img src="https://render.githubusercontent.com/render/math?math=\frac{2}{\beta N}KL(q_\theta(z|x)\|p(z))=\frac{1}{\beta N}\sum_{j=1}^J(-1-\log{\sigma_{j}^2}%2B\mu_{j}^2%2B\sigma_{j}^2)=\lambda\sum_{j=1}^J(-1-\log{\sigma_{j}^2}%2B\mu_{j}^2%2B\sigma_{j}^2)">, where <img src="https://render.githubusercontent.com/render/math?math=\lambda=\frac{1}{\beta N}">.

### Final Form of the Loss Function

Altogether, for each data point in the training set, the loss function is

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}=\frac{1}{L}\sum_{l=1}^L[\frac{1}{N}\sum_{n=1}^N(x_{(n)}-x_{\phi,(n)})^2]%2B\lambda\sum_{j=1}^J(-1-\log{\sigma_{j}^2}%2B\mu_{j}^2%2B\sigma_{j}^2)">

This can be averaged over a mini-batch to obtain the overall loss function.


[^1]: Diederik P Kingma, Max Welling: “Auto-Encoding Variational Bayes”, 2013; <a href='http://arxiv.org/abs/1312.6114'>arXiv:1312.6114</a>.

