# Learning Differentiable Programs with Admissible Neural Heuristics

## Requirements

- Python 3.8.1+

- PyTorch 1.4.0+

- scikit-learn 0.22.1+

## Code Structure

- `algorithms/` contains seperate files for each program learning algorithm (easy to add more).

- `dsl/` contains our library functions. The default functions are `dsl/library_functions.py`. We've included additional files for domain specific functions, such as `dsl/running_averages.py` and `dsl/crim13.py`.

- `utils/` contains helper functions, sorted by functionality.

- `program_graph.py` contains all functions operating on the graph of programs.

- `dsl_current.py` defines the DSL and any custom edge costs for the graph. This file can be changed for different experiments.

- `train.py` is the command-line script for running a program learning algorithm for a sequence classification task.

- `eval.py` is the command-line script for evaluating a program on the test set.

## Running the Code

### Train

For running aprogram learning algorithm, use `train.py` and command-line arguments to define the algorithm, data, and any hyperparameters.
### Eval

To evaluate a program on a test set, use `eval.py` and use command-line arguments to specify the program and test set.

### Examples

Examples are in the bash script `commands_example.sh`, which contains training and eval scripts on a toy dataset in `data/example/`. You can copy the commands onto your command-line or run the entire script with `$ ./commands_example.sh` (should be fast for toy dataset).

## Adding new data

To add a new dataset, you need to specify the path to data with the following command-line arguments: `--train_data`, `--test_data`, `--train_labels`, `--test_labels`.  You can also specify validation set paths using `--valid_data` and `--valid_labels`. If the validation set is not specified, then a split from the training set is used for validation.

Furthermore, be sure to specify `--input_type`, `--output_type`, `--input_size`, `--output_size`, and `--num_labels`.

### CRIM13

The processed files for the CRIM13 dataset are provided the following anonymized link: https://drive.google.com/drive/folders/1vV2p_g-Y1yKdGOtAVzBnWnZdDWhbqjyb?usp=sharing. To use this data, follow these steps:

1) Download the files in the link above into `data/crim13_processed/`.

2) We also provide the CRIM13 DSL we used in `dsl_crim13.py`. To use this DSL, import DSL_OBJECT and CUSTOM_EDGE_COSTS from `dsl_crim13.py` instead, at the top of `train.py`.

3) Run the commands in `commands_crim13.sh`.

## Adding new DSL functions

To add new domain-specific library functions, follow these steps:

1) Create a new file in `dsl/` with the functions. See `dsl/running_averages.py` and `dsl/crim13.py` as examples.

2) Update `dsl/__init__.py` to import your functions.

3) Update DSL_DICT in `dsl_current.py` with the functions you want to be a part of the graph.

4) (optional) Update CUSTOM_EDGE_COSTS `dsl_current.py` with custom costs. Default costs will be the number of neural modules in the function.

5) If any function has any additional hyperparamters (e.g. beta paramter for ITE), you need to update `construct_candidate()` in `program_graph.py`.

6) If you happen to define a ListToListModule or a ListToAtomModule without any neural modules (i.e. program can become fully symbolic when replacing with such function), you need to update `min_depth2go()` in `program_graph.py`.
