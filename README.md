<!-- ABOUT THE PROJECT -->
## Tabular Data Generator

Tabular data generator model based on the integration of Gaussian Mixture model and Variational Autoencoders. 

This repository provides:
* Necessary scripts to train the generator and the state-of-the-art model CTGAN using [SDV](https://github.com/sdv-dev/SDV) package.
* Preprocessing for Adult, METABRIC and STD.
* Validation techniques: RF, SDV technique and utility validation (classification or survival analysis [Coxph](https://github.com/havakv/pycox)).
* Saved evaluation summaries for the three datasets in the paper.

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

### Show the manuscript results

Install the dependencies in `requirements.txt` and run these commands from
the repository root.

```sh
python main_generator.py
python main_sota.py
```

In PyCharm, select the Python 3.11 interpreter with those dependencies and run
either script directly. The generator displays VAE-BGM and VAE-Gauss. The
second script displays CTGAN.

The default configuration loads the nine saved `results.pkl` files under
`results/manuscript/` for Adult, METABRIC and STD. Both training and evaluation
are disabled in `utils.py`. For VAE-BGM and VAE-Gauss, each summary reports
the three seeds with the lowest Random Forest discrimination accuracy out of
the 15 saved seeds. The displayed metrics are discrimination accuracy,
statistical quality and downstream utility.

### Train and evaluate the original model

In `utils.py`, set `args['train']` and `args['eval']` to `True` and run
`main_generator.py`. The run uses Adult, METABRIC and STD, 15 seeds, a batch
size of 500 and up to 1,000 epochs with early stopping. It writes to
`results/output_generator_paper_full_runs/`. Run `main_sota.py` afterwards
to train and evaluate CTGAN using the prepared data. CTGAN writes to
`results/output_sota_runs/`.

The manuscript summaries are included in Git. Run outputs such as weights,
synthetic datasets and figures are generated locally.

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

