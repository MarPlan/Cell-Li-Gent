## Overview
Ramping up battery modeling by more intensively using data, cutting down on the need for tailored testing protocols typically used to identify parameters. Instead, a randomized protocol helps create synthetic datasets through electrochemical simulation to train a neural network. Building from my thesis proposal, the target is to develop a model that predicts and also reveals new, efficient ways to operate batteries using an agent.

My [thesis](/doc/thesis.pdf) benchmark indicates that neural networks with few parameters (~350k) can already perform on an order of magnitude comparable to traditional methods.

| Dataset | Model                         | MSE    | MAE    |
|---------|-------------------------------|--------|--------|
| LFP     | Data-Driven (Transformer)     | 7.3e-4 | 2.0e-2 |
| LFP     | Classical (ECM)               | 3.5e-4 | 1.5e-2 |
| NMC     | Data-Driven (Transformer)     | 8.3e-5 | 8.0e-3 |
| NMC     | Classical (ECM)               | 1.0e-7 | 2.9e-4 |

## Structure

## Results
