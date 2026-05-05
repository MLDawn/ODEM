
# Implementation of a Online Generalised Predictive Coding through Online Dynamic Expectation Maximisation (ODEM)
This is a Python/Pytorch package, where the neuronal message-passing in a one-layer Predictive Coding (PC) network has
implemented using Python and Pytorch running on CPU.
---
<p align="center">
  <img src="example/lorenz-GM-kx=3.png" alt="Lorenz GM kx=3" width="600"/>
  <br>
  <em>Figure 1: State estimation using a Lorenz generative model vs. a Generalised Lotka-Volterra generative process (i.e., under model mismatch) using ODEM with 3 orders of generalised coordinates of motion. The inferred trajectory closely tracks the true latent dynamics despite structural mismatch between the generative model and process.</em>
</p>

---
# Installing required packages
Go to the directory of your choice where you would like to clone the repository:
```commandline
cd MY_DIRECTORY
```
Then clone the repository on your machine:
```commandline
git clone https://github.com/MLDawn/ODEM.git
```
Go inside the cloned repository:

```commandline
cd ODEM
```
Create a conda environment with your name of choice, **_ENVIRONMENT_NAME_**, with Python version 3.11.5:
```commandline
conda create -n ENVIRONMENT_NAME python == 3.13.5 
```
Activate the created environment:
```commandline
conda activate ENVIRONMENT_NAME
```
Using the provided _**requirements.txt**_ file, install all the necessary packages (This will take a while):
```commandline
pip install -r requirements.txt 
```
---
# Running the code
1 - Open the **_parameters.yaml_** file and set your configuration for ODEM.

2 - Open the project in an IDE (e.g., Pycharm).

3 - Assign **_ENVIRONMENT_NAME_** as the Python interpreter to this project in your IDE.

4 - Run **_main.py_**.