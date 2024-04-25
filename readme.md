# Physics-Informed Graph Neural Cellular Automata: an Application to Compartmental Modelling

## Installation
The software runs on Python 3.8.10 with CUDA 11.6.2 and compatible Nvidia drivers. The framework used is Torch 1.12.1+cu116 and Torch Graph 2.1.0.post1. <br />
The container.def file contains instructions for generating a container that runs on Singularity. Instead, requirements.txt contains the versions of various Python packages and is generated with pip freeze.

## Datasets
The datasets are generated using the script data/SIR_simulation_graph and by calling the function simulate_all_iterations(b, g, points, seed), where b is beta, g is gamma, points indicate the total length of the dataset, and seed relates to the initialization of the simulation. The graph structure is loaded from files.

By calling the function gen_inputs(seeds, params) from SIR_simulation_graph.py, a dataset is generated in the dataset1/in folder with a different number of seeds. For example, to generate 20 datasets of 275 points each, you call gen_inputs(20, {"test_points":275}).

In dataset1/in, datasets with Erdős–Rényi structure are present, while in dataset2/in, Barabási–Albert datasets are present. The thesis tests are conducted using dataset1, so for dataset2, one might not expect the hyperparameters to be correctly set.

## Experiments
Under dataset1/, there are several folders, one for each type of experiment, identified as follows: <br />
(sindy/recurrent)_(node/aggr)_( _ /no_constr)_(trainPoints) <br />
recurrent indicates the GNNs, (node/aggr) indicates the type of supervision, if nothing is indicated about the constraints, it means they are present, and the number of points for training can be 100 or 150.

For each type of experiment, there is a parameters.json file containing various hyperparameters. <br />

For Sindy, there is a parameter "dictionary" containing the various functions in the dictionary; this is only used to ensure that the functions contained in the implementation are the desired ones. If you want to modify the functions to use, you also need to change them in the __init__ method of the Sindy class in the sindy_graph.py file. <br />
"hyper_parameters" contains a list of hyperparameter ranges. For Sindy, the first hyperparameter is unused, the second indicates the learning rate as a range, i.e., [start, end, step]. For GNNs, the first hyperparameter indicates the learning rate, the second the number of neurons per layer, and the third the number of dense layers. <br />
"dual_rate_multiplier" contains the learning rate multiplication factor to set the lambdas in case constraints are used, and thus the method of Lagrange multipliers. In fact, the learning rate for lambdas is calculated as dual_rate = dual_rate_multiplier * learning_rate. <br />
The "plots" parameter indicates whether to show the various plots at the end of the training of a dataset (1 seed). To change the plots to display, you need to modify the grid_search() function contained in the sindy_graph.py and graph_recurrent.py files, respectively.
"coeffs_increment" indicates what fraction of the total number of coefficients to remove at each Sindy thresholding step.
"tolerance_epochs" indicates the tolerance to use for early stopping.
In Sindy, the number of coefficients to remove is determined by the validation error. To do this, early stopping is used with "tolerance_thresholds".
"bootstrap_increment" is used with aggregated supervision to determine by how many timesteps to increase the supervision window at each training phase.
To run the experiments, simply call python3 sindy_graph.py for Sindy or python3 graph_recurrent.py for GNNs. In both cases, the parameters are <br />
[--first_seed FIRST_SEED] [--last_seed LAST_SEED] [--dataset DATASET] [--test_name TEST_NAME] <br /> an example is <br />
python3 sindy_graph --first_seed 0 --last_seed 19 --dataset 'dataset1' --test_name 'sindy_aggr_150' <br />
the command should be executed from the /scripts folder. In test_name, indicate the name of the folder under /dataset1 containing the parameters.json file of the experiment to run. As the experiment runs, the folder will be populated with seed_model.data files containing the model found for each seed, seed_out_str.data with the terminal output from the experiment, and for Sindy, also the seed_active.data file with the coefficients present after the thresholding procedure.

Once the experiment is finished for all seeds, it is possible to calculate the aggregate statistics of all seeds by respectively going into the sindy_graph.py or graph_recurrent.py file and in main() commenting out the line with graph_utils.test_many_seeds and uncommenting graph_utils.many_seeds_stats, then simply run the script with the same command used for training (be careful that the range of seeds must correspond to the models actually saved to file). If you also want to view the plots, simply set the "plots" parameter in the respective configuration file. <br />
It is also possible to run tests by setting the default parameters in the scripts and calling only python3 sindy_graph.py, this simplifies setting up a debugger.





