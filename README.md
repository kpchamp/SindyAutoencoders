# SindyAutoencoders

Code for the paper ["Data-driven discovery of coordinates and governing equations"](https://arxiv.org/abs/1904.02107) by Kathleen Champion, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton.

The paper contains results for three example problems. The results in the paper were obtained by running ten experiments on each example system using the parameters found in the provided code. Our training procedure can be replicated by running the scripts train_lorenz.py, train_reactiondiffusion.py, and train_pendulum.py. Each of these scripts will run ten instances of the training and produce ten models. Each script will also produce summary output, saved in a Pandas dataframe. The information saved in the dataframe can be used to compare among models. In the paper, we perform a model selection among the resulting models and select a subset of the models to highlight. Our model selection procedure is described in the appendix, along with a detailed description of the training procedure.

Saved models can be loaded using the script load_saved_model.py. This script takes two arguments: the example system ('lorenz', 'pendulum', or 'reactiondiffusion') and the base filename of the model to load. The script will generate a set of test data for the example system, load the TensorFlow model, and apply the model to the data.
