# 2d-mps

Code implementing the Method of Particular Solutions from "Reviving the Method
of Particular Solutions" by Timo Betcke and Lloyd N. Trefethen ([SIAM
2005](https://doi.org/10.1137/S0036144503437336)).

## Instructions to run

### Base Python (recommended)

First, if your command prompt says `(base)` at the left of the terminal window,
this indicates that Anaconda/Miniconda is running. If so, run

```sh
$ conda deactivate
```

to ensure that Conda will not interfere with Python's package manager.

Next, run the following to create a Python virtual environment specific to this
project:

```sh
$ python3 -m venv venv
```

Then, activate the environment with

```sh
$ source venv/bin/activate
```

Finally, install the needed packages with

```sh
$ pip install -r requirements.txt
```

Now, you can run any script using

```sh
$ python script_to_run.py
```

After this initial setup, you can come back to the project and run any script
by first reactivating the environment and then invoking the `python` command as
shown above (there is no need to recreate the virtual environment or install
packages again).

### Anaconda/Miniconda

If you are running Anaconda/Miniconda and would rather run this code in the
context of Conda, ensure that `numpy` and `scipy` are installed at a minimum.
Additionally, `matplotlib` is needed for plotting, and the decagon code uses
`pandas`. Lastly, it can be helpful to use Jupyter Notebooks, in which case
`jupyterlab` is needed. All five packages can be installed with

```sh
$ conda install numpy scipy matplotlib pandas jupyterlab
```

After the needed packages are installed, you can run any script using

```sh
$ python script_to_run.py
```
