# PHYS449

## Dependencies

- json
- numpy
- random
- argparse
- matplotlib

## Running `main.py`

Note that `param.json` should be placed in the same directory as `main.py`, as well as the `data` folder, which should contain `in.txt`. To run `main.py`, use

```sh
python main.py data/in.txt
```
The program generates `KL_div_during_training.pdf` in the same directory.

## json parameters
- Number of gd iterations: number of iterations of gradient descent during training
- learning rate: scale factor for each step of the gradient descent
- Number of iterations per MC simulation: number of steps in each Monte-Carlo simulation
- Number of MC simulations: total number of Monte-Carlo simulations performed
- verbosity: when the verbosity is `2` (the value in `param.json`), the program tracks the KL divergence of the
training dataset with respect to your generative model during the training and
save a plot of its values versus training epochs

## Mathematical Details
### Calculation of the KL divergence

To calculate the KL divergence <img src="https://render.githubusercontent.com/render/math?math=\mathrm{KL}(p|p_\lambda)=\sum_{x\in\Omega}p(x)\log{\frac{p(x)}{p_\lambda(x)}}=\sum_{x\in\Omega}p(x)\log{p(x)}-\sum_{x\in\Omega}p(x)\log{p_\lambda(x)}\approx\sum_{x\in\Omega}p_D(x)\log{p_D(x)}-\sum_{x\in\Omega}p_D(x)\log{p_\lambda(x)}">, we use the distribution <img src="https://render.githubusercontent.com/render/math?math=p_\lambda(x)"> obtained from the Monte-Carlo simulations. Note that it is possible that for <img src="https://render.githubusercontent.com/render/math?math=x\in\Omega"> such that <img src="https://render.githubusercontent.com/render/math?math=p_\lambda(x)\approx0">, it is possible that it does not show up in the Monte-Carlo simulations at all, then the distribution would predict <img src="https://render.githubusercontent.com/render/math?math=p_\lambda(x)=0">. If so, <img src="https://render.githubusercontent.com/render/math?math=p_D(x)\log{p_\lambda(x)}">
becomes undefined.

However, in that case, since <img src="https://render.githubusercontent.com/render/math?math=p_D(x)\approx p_\lambda(x)\approx0"> and <img src="https://render.githubusercontent.com/render/math?math=\lim_{p\rightarrow0}p\log{p}=0">, we have <img src="https://render.githubusercontent.com/render/math?math=p_D(x)\log{p_\lambda(x)}\approx p_D(x)\log{p_D(x)}\approx0">. ie we can ignore the contributions from these terms. Specifically, <img src="https://render.githubusercontent.com/render/math?math=p_D(x)\log{p_\lambda(x)}\approx p_D(x)\log{p_D(x)}"> cancels out the corresponding term in <img src="https://render.githubusercontent.com/render/math?math=\sum_{x\in\Omega}p_D(x)\log{p_D(x)}">.

(If we ignore the <img src="https://render.githubusercontent.com/render/math?math=p_D(x)\log{p_\lambda(x)}"> term instead of doing a cancellation, then the KL-divergence becomes negative at times.)

### <img src="https://render.githubusercontent.com/render/math?math=L^2"> regularizer

We use an <img src="https://render.githubusercontent.com/render/math?math=L^2"> regularizer, ie an additional <img src="https://render.githubusercontent.com/render/math?math=\lambda\sum_{i=0}^n w_{i}^2"> term in the loss fuction. Its effect in the update rule for gradient descent for <img src="https://render.githubusercontent.com/render/math?math=w_j"> is an additional term <img src="https://render.githubusercontent.com/render/math?math=-\frac{\partial}{\partial w_j}(\lambda\sum_{i=0}^n w_{i}^2)=-2\lambda w_j">.
