<!-- ABOUT THE PROJECT -->
## Tabular Data Generator

Tabular data generator model based on the integration of Gaussian Mixture model and Variational Autoencoders. 

This repository provides:
* Necessary scripts to train the generator and the state-of-the-art model CTGAN using [SDV](https://github.com/sdv-dev/SDV) package.
* Pre-processed and ready-to-use datasets included.
* Validation techniques: RF, SDV technique and utility validation (classification or survival analysis [Coxph](https://github.com/havakv/pycox)).
* Pre-trained models to save you training time for std and metabric (adult weighted too much to upload it to the repo).

For more details, see full paper [TBC]().


<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:

* Ubuntu
* Python 3.8.0
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


To train/test the proposed model and show results, run the following command:
   ```sh
   python data_generation/main_generator.py
   ```
To train/test SOTA models and show results, run the following  (it is necessary to run first the proposed model to generate the datasets for the SOTA model to train on them)
   ```sh
   python data_generation/main_sota.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



[//]: # (<!-- LICENSE -->)

[//]: # (## License)

[//]: # ()
[//]: # (Distributed under the XXX License. See `LICENSE.txt` for more information.)

[//]: # ()
[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



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

