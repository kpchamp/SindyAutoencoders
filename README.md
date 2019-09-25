# SindyAutoencoders

Code for the paper ["Data-driven discovery of coordinates and governing equations"](https://arxiv.org/abs/1904.02107) by Kathleen Champion, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton.

The paper contains results for three example problems based on the Lorenz system, a reaction-diffusion system, and the nonlinear pendulum. Code for each example can be found in the respective subfolder in the examples folder. For each example, there are jupyter notebooks for (1) running the training procedure and (2) analyzing the resulting models.

Our training procedure can be replicated using the jupyter notebooks in each folder, which will run ten instances of the training and produce ten models. Running the notebook will also produce summary output, saved in a Pandas dataframe. The information saved in the dataframe can be used to compare among models. In the paper, we perform a model selection among the resulting models and select a subset of the models to highlight. Our model selection procedure is described in the appendix, along with a detailed description of the training procedure.

For each example, we also include jupyter notebooks to analyze the resulting models. These notebooks produce plots of the results and print out summary statistics on test data. For the Lorenz and pendulum example problems, the models analyzed in the paper are included in the repository. Due to large file size, the reaction-diffusion models are not included in the repository.
