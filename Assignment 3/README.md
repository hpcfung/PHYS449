# PHYS449

## Dependencies

- numpy
- matplotlib
- torch
- json
- argparse
- sys

## Running `main.py`

Note that `param.json` should be placed inside the `param` folder. To run `main.py`, use

```sh
python main.py [--param param.json] [-v N] [--res-path results] [--x-field x**2] [--y-field y**2] [--lb LB] [--ub UB] [--n-tests N_TESTS]
```
eg
```sh
python main.py --param param/param.json -v 2 --res-path plots --x-field "-y/np.sqrt(x**2 + y**2)" --y-field "x/np.sqrt(x**2 + y**2)" --lb -1.0 --ub 1.0 --n-tests 3
```

or
```sh
python main.py --param param/param.json -v 2 --res-path plots --x-field "np.sin(np.pi*x) + np.sin(np.pi*y)" --y-field "np.cos(np.pi*y)" --lb -1.0 --ub 1.0 --n-tests 3
```

Note that the output plot is displayed immediately and is also saved as a pdf.
## json arguments

ode
- T: the neural network solves the DE from t = 0 to t = T
- solution steps: the number of steps used when generating the solutions
--------------------------------------------------------------------------------------------------------------------------------------
scale factors
- training grid scale factor: when generating the training dataset, we consider solutions with initial conditions beyond the given bounds in case solution leaves the boundary; scale the given bounds by this factor
- plot scale factor: plot a region larger than the initial conditions bounds (by this factor)
--------------------------------------------------------------------------------------------------------------------------------------
grid sizes
- training grid size: the initial conditions for the training set are obtained on an n x n grid
- plot grid size: plot the vector field on an n x n grid
--------------------------------------------------------------------------------------------------------------------------------------
optim
- number of epochs: number of epochs used when training the neural network
- learning rate: learning rate used in the Adam optimizer
- batch size: batch size used in the dataloader
--------------------------------------------------------------------------------------------------------------------------------------
model
- hidden layer 1 width: number of neurons in the first hidden layer
- hidden layer 2 width: number of neurons in the second hidden layer
- hidden layer 3 width: number of neurons in the third hidden layer
--------------------------------------------------------------------------------------------------------------------------------------
testing
- number of points tested: number of ICs tested when calculating the testing loss
