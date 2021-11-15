# PHYS449

## Dependencies

- json
- numpy
- random
- argparse

## Running `main.py`

To run `main.py`, use

```sh
python main.py data/in.txt
```

```sh
cd C:\Python_projects\PHYS 449\HW4
```
```sh
python MCMC_gd_argparse.py data/in.txt
```
```sh
python MCMC_gd_argparse.py -h
```
## json arguments
- Number of gd iterations: number of iterations of gradient descent during training
- learning rate: scale factor for each step of the gradient descent
- Number of iterations per MC simulation: number of steps in each Monte-Carlo simulation
- Number of MC simulations: total number of Monte-Carlo simulations performed
