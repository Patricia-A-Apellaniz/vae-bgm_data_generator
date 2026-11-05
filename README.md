<!-- ABOUT THE PROJECT -->
## Tabular Data Generator

Tabular data generator model based on the integration of Gaussian Mixture model and Variational Autoencoders. 

This repository provides:
* Necessary scripts to train the generator and the state-of-the-art model CTGAN using [SDV](https://github.com/sdv-dev/SDV) package.
* Pre-processed and ready-to-use datasets included.
* Validation techniques: RF, SDV technique and utility validation (classification or survival analysis [Coxph](https://github.com/havakv/pycox)).
* Pre-trained models to save you training time for std and metabric (adult weighted too much to upload it to the repo).

For more details, see full paper [here](https://ieeexplore.ieee.org/document/10715230).


<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:

* Python 3.11 (the original paper environment used Python 3.8)
* Windows or Linux
* Packages in requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

### Installation

Download the repo manually (as a .zip file) or clone it using Git.
   ```sh
   git clone https://github.com/Patricia-A-Apellaniz/vae-bgm_data_generator
   ```


<!-- USAGE EXAMPLES -->
## Usage

You can specify different configurations or training parameters in utils.py for both models.

The default model profile is `paper`. It retains the original VAE architecture,
loss and minibatch validation behavior. `utils.py` selects the three paper
datasets. Training and evaluation are enabled in the current configuration.
New runs go to `data_generation/output_generator_runs/`. Both VAE-BGM and
VAE-Gauss are enabled. CTGAN runs go to `data_generation/output_sota_runs/`.
MMD is optional and disabled by default.

### Reproduce and compare model variants

From the repository root, run:

```sh
python -m data_generation.benchmark --profiles paper --seeds 0 1 2 --epochs 1000
python -m data_generation.benchmark --profiles training stability decoder --seeds 0 1 2 --epochs 1000
python -m data_generation.benchmark --summary
```

The benchmark uses the existing dataset preprocessing, generation, RF
discriminative validation, SDMetrics quality and task utility (classification
or CoxPH). Each run saves its weights, parameters and metrics under
`experiments/benchmark/`. It reuses completed training and skips completed
evaluations. Use a different `--output` directory when changing the epoch budget
or experiment settings. `--reevaluate` repeats generation and evaluation from
the saved VAE weights with the selected seed. Dataset downloads require an Internet connection on
the first run.

Available profiles are `paper`, `training` (one validation per epoch and restore
best validation weights), `stability` (also gradient clipping and latent
log-variance bounds), `warmup` (also 50 epochs of KL warmup), `decoder` (stability
plus softplus Gaussian scales and zero dropout), and `posterior` (BGM fitted on
five posterior samples per training row). These are experimental alternatives.
Comparisons require multiple seeds and all three datasets before choosing a
new default. Lower RF distinguishability and higher quality/utility are better.
Running the pipeline successfully does not establish reproduction of the
paper's numerical results with a different software environment.

On 2026-11-05, all six profiles completed on all three datasets with seeds
0, 1 and 2, a 1,000-epoch limit and early stopping. The environment was Python
3.11.15, PyTorch 2.15.0 CPU and the pinned NumPy, pandas, scikit-learn, SDV and
SciPy versions in `requirements.txt`. The paper preprocessing and evaluation
were retained. The following values are means over three seeds. Utility is
classification accuracy for Adult and CoxPH concordance for METABRIC and STD.

| Dataset | Profile | RF accuracy (lower is better) | Quality | Synthetic utility |
|---|---|---:|---:|---:|
| Adult | paper | 0.7244 | 0.9234 | 0.7794 |
| Adult | decoder | 0.6241 | 0.9616 | 0.8038 |
| Adult | warmup | 0.6928 | 0.9322 | 0.7765 |
| METABRIC | paper | 0.7618 | 0.8387 | 0.5778 |
| METABRIC | decoder | 0.6142 | 0.8574 | 0.5861 |
| METABRIC | warmup | 0.7022 | 0.9088 | 0.5860 |
| STD | paper | 0.6679 | 0.9005 | 0.5703 |
| STD | decoder | 0.6749 | 0.9283 | 0.6013 |
| STD | warmup | 0.6334 | 0.9000 | 0.6149 |

The decoder profile improves all three means on Adult and METABRIC. STD shows
a tradeoff between RF distinguishability, quality and utility. Three seeds
provide an exploratory comparison. The default remains `paper` for
reproducibility. Per-seed results for every profile are in
[`benchmarks/comparison_2026-11-05.csv`](benchmarks/comparison_2026-11-05.csv).
The two main scripts also completed a short STD run with VAE-Gauss and MMD,
and CTGAN completed 20-epoch smoke runs on all three datasets. Eight automated
tests cover validation frequency, best-weight restoration, latent bounds,
preprocessing, MMD and CLI save/load reproducibility. The original VAE and the
default profile produced identical losses and weights in a controlled
two-epoch check against commit `6a62495`.

### Use your own CSV

Use one row per observation and one column per feature. Remove identifiers and
split off any final test set before passing the CSV to this command.

```sh
python -m data_generation.external train --input my_data.csv --output external_output --epochs 1000 --samples 1000
python -m data_generation.external sample --model external_output --output new_samples.csv --samples 500 --seed 1
```

Numeric columns default to Gaussian variables. Text columns default to
categorical variables. For integer categories, binary variables or another
explicit assignment, supply `--schema schema.json` with a JSON mapping such as:

```json
{"age": "gaussian", "sex": "categorical", "outcome": "bernoulli"}
```

The supported distributions are `gaussian`, `categorical` and `bernoulli`.
Bernoulli columns must have two observed training categories. Training and
validation are split before fitting the scaler, category vocabulary and
imputation. Unseen validation categories are masked. Missing cells are imputed
and masked in the loss, and generated CSVs contain complete rows. Date handling,
domain constraints and task-specific utility require dataset-specific logic.

The output directory contains `synthetic.csv`, training history, preprocessing
metadata, VAE weights and the fitted BGM. The sample command loads pickle for
the scikit-learn BGM, so use trusted model directories. `--profile stable`
enables the optional training and latent stability changes. Use `--help` for
dimensions, batch size, learning rate and validation fraction.


To train/test the proposed model and show results, run the following command:
   ```sh
   python data_generation/main_generator.py
   ```
To train/test SOTA models and show results, run the following  (it is necessary to run first the proposed model to generate the datasets for the SOTA model to train on them)
   ```sh
   python data_generation/main_sota.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->

## License


Distributed under the MIT License. See `LICENSE` for more information.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Patricia A. Apellaniz - patricia.alonsod@upm.es

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[//]: # (<!-- ACKNOWLEDGMENTS -->)

[//]: # (## Acknowledgments)

[//]: # ()
[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)

