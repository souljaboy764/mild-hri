# MILD-HRI

Exploring the use of HSMMs for learning latent space dynamics in VAEs

## Data Preprocessing

All the preprocessed data is present in the [`data`](data/) folder. The following instructions are on how to perform them from scratch.

For the Buetepage dataset, extract the dataset from `https://github.com/souljaboy764/human_robot_interaction_data/` to a location on your PC and run the preprocessing according to the repository `buetepage_phri` and use the file `traj_data.npz` as input. This is done individually for the HHI and HRI scenarios.

For the NuiSI dataset, see the instructions in `https://github.com/souljaboy764/nuisi-dataset` for further information on preprocessing.

## Prerequisites

This repo uses the pytorch version of pbdlib at `https://git.ias.informatik.tu-darmstadt.de/prasad/pbdlib-torch` for the HMMs. Other requirements are give in [`requirements.txt`](requirements.txt)

## Training

To train the networks, first train the model Human-Human Interaction data with [`mild_hri/train_hh.py`](mild_hri/train_hh.py) and then use the learned model to train on the Human-Robot Interaction data with [`mild_hri/train_hr.py`](mild_hri/train_hr.py). Some sample configurations for training can be found in [`mild_hri/train.sh`](mild_hri/train.sh) for both HHI and HRI.

## Testing

For evaluating the Mean Squared Error, run the file [`test.py`](test/test.py) after editing the checkpoints in the code. The main evaluation is done by the `evaluate_ckpt_hh` and `evaluate_ckpt_hr` functions in [`mild_hri/utils.py`](mild_hri/utils.py) which are called on the test data.

## User study evaluation

The evaluation of the user study responses are found in [`test/data_eval.ipynb`](test/data_eval.ipynb) and the corresponding parameters to plot can be set in the initial block.