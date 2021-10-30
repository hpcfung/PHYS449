# PHYS449

## Dependencies

- json
- numpy

## Running `main.py`

Note that `param.json` should be placed inside the `param` folder. To run `main.py`, use

```sh
python main.py
```
Note that the output plot is displayed immediately and is also saved as a pdf.
## json arguments

- T: the neural network solves the DE from t = 0 to t = T
- solution steps: the number of steps used when generating the training dataset
- step size in training data: choose a subset of the data generated as the training set (eg if step size = 25, we choose the 25th, 50th, 75th, etc)
-----------------------------------------------------------------------------------------------------------------------------------------
- IC bounds scale factor: when generating the training dataset, we consider solutions with initial conditions beyond the given bounds to improve the behavior of solutions near the boundary; scale the given bounds by this factor
- plot scale factor: plot a region larger than the initial conditions bounds (by this factor)
-----------------------------------------------------------------------------------------------------------------------------------------
- plot grid size: plot the vector field on an n x n grid
- IC grid size: the initial conditions for the training set are obtained from this n x n grid
