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

Since we want to maximize the likelihood <img src="https://render.githubusercontent.com/render/math?math=p(x)">, we maximize the lower bound <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]-KL(q_\theta(z|x)\|p(z))">. Hence the loss function, the function we minimize during training, is <img src="https://render.githubusercontent.com/render/math?math=L=-\mathbb{E}_\epsilon[\log{p_\phi(x|z=\mu_\theta(x)%2B\sigma_\theta(x)\odot\epsilon)}]%2BKL(q_\theta(z|x)\|p(z))">. These two terms can be further simplified.


sth[^1]


[^1]: Diederik P Kingma, Max Welling: “Auto-Encoding Variational Bayes”, 2013; <a href='http://arxiv.org/abs/1312.6114'>arXiv:1312.6114</a>.

