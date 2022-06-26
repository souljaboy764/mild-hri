# HSMMVAE

Exploring the use of HSMMs for learning latent space dynamics in VAEs

## Datsets

TODO:

- Structure all the preproc to save the datasets as a list with `A` entries (one for each action)
- Each entry will have `N` trajectories.
- Each trajectory will have `T_N` number of time steps
- Each timestep has `D` degrees of freedom.
- If each action has the same number of demonstrations which are sampled to have the same length, then the dataset would be a numpy array of size (A, N, T, D)
