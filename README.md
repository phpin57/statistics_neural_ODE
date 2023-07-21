# statistics_neural_ODE

There a 3 different files in this repo related to the project :

1. **Equa diff.ipynb** : this Jupyter Notebook is used to represent all graphical visualizations in the diagonal case. In particular, the code related to the two plots of the report can be found at the very end of the notebook

2. **calcul a optimal.py** : it is a small Python file used to calculate in which interval ]0,a[ we have exponential stability

3. **ode_example.py** : this Python file is used for the general linear case and takes part of the code from the package torchdiffeq to adapt it to our problem. It trains an unbiased linear layer to learn a specific linear mapping, and plots both the frobenius loss as well as the evolution of the exponential eigenvalues. 
