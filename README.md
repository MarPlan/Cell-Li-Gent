<!--- # Cell-Li-Gent--->
## Overview

This project emphasizes a data-centric approach to enhance battery modeling, aiming to improve the accuracy of state estimation by utilizing the datas full potential. Synthetic datasets, generated through electrochemical methods, inform the training of specialized neural networks. Initial benchmarks provide a framework for this exploration. The aspiration is to develop a model that offers non-conventional outputs, potentially facilitating innovative operational strategies through the deployment of a more capable learning agent.

## Motivation

This project draws from foundational concepts initially laid out in my [master’s thesis](/inspiration/thesis.pdf) proposal, setting the stage for their expansion and broader application in battery modeling:

| Dataset | Model                         | MSE    | MAE    |
|---------|-------------------------------|--------|--------|
| LFP     | Data-Driven (Transformer)     | 7.3e-4 | 2.0e-2 |
| LFP     | Classical (ECM)               | 3.5e-4 | 1.5e-2 |
| NMC     | Data-Driven (Transformer)     | 8.3e-5 | 8.0e-3 |
| NMC     | Classical (ECM)               | 1.0e-7 | 2.9e-4 |

This table showcases the best-performing models for State-of-Charge (SoC) prediction as documented in the thesis. Focusing (order of magnitude) on accuracy of compact, data-driven models (350k parameters) in contrast to traditional Equivalent-Circuit-Models (ECMs).

## Results
...to be continued...
