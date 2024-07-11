# Cell-Li-Gent

A data-driven approach for multivariate time series prediction, aiming
to simplify the battery modeling process and cut down on expensive
battery tests. Using a randomized protocol to create a synthetic dataset
via electrochemical simulation (SPMe from
[PyBaMM](https://github.com/pybamm-team/PyBaMM) \[1\]). This dataset
then serves as fuel to train the neural networks. Extending what I laid
out in my [thesis](/doc/thesis.pdf) proposal, the goal is to not just
predict battery behavior but to also uncover more efficient ways to
operate batteries using an intelligent agent.

Here’s a snapshot from my [thesis](/doc/thesis.pdf) where I compared a
classic method against a neural network that doesn’t go overboard on the
parameters (~350k) but still delivers performance that’s pretty much on
par with the traditional model.

| Dataset | Battery-Model             | MSE    | MAE    |
|---------|---------------------------|--------|--------|
| LFP     | Data-Driven (Transformer) | 7.3e-4 | 2.0e-2 |
| LFP     | Classical (ECM)           | 3.5e-4 | 1.5e-2 |
| NMC     | Data-Driven (Transformer) | 8.3e-5 | 8.0e-3 |
| NMC     | Classical (ECM)           | 1.0e-7 | 2.9e-4 |

To be fair, the ECM (w/o Kalman-Filter) had as input only the current
values where the neural network got current, terminal voltage and
surface temperature. The predicted output was State-of-Charge,
respectively.

## Overview

The [model](/model) directory has the designs I’m considering:

- Transformer based on Meta’s [Llama
  3](https://github.com/meta-llama/llama3) \[2\]

- Mamba (State-Space) based on
  [Mamba](https://github.com/state-spaces/mamba) **Gu2023?**

This project also heavily utilizes Andrej Karpathy’s
[nanoGPT](https://github.com/karpathy/nanoGPT/tree/master).

Why am I picking these architectures for battery state estimation? It’s
because Transformers are everywhere and set the baselines. The
State-Space method has been reliable in traditional battery modeling by
considering historical information. Also, both architectures performed
well in my thesis benchmark.

There’s a lot more interesting architectures and combinations
thereof—hopefully, as computational costs might distribute, digging into
these will be feasible.

### Randomized Data Profiles

Snapshot of the input space for the neural network, based on the
parameters what are measured in a real world scenario:

- Terminal Current (generated randomly)
- Terminal Voltage
- Surface Temperature

The current profiles come in three different variations: first is a step
profile (blue). The second (green) combines charging and discharging
phases from a drive cycle and tailored to the cell. The third one (red)
is a superimposed version of the both, smoothed out with a Gaussian
kernel. These profiles are supposed to cover static and dynamic aspects.
![profiles](./doc/profiles_plot.png) Throughout the entire sequence, the
profiles are designed to stay within the specifications of the datasheet
\[3\]. For the seak of simplicity I am focusing on short-term current
(10s), continuous current, and critical current specific to SoC ranges
(SoC\<0.2, SoC\>0.8). A single profile guarantees one full charge and
discharge, with a zero current intervals for equilibrium information.
![profiles3d](./doc/data_3d.png) The plane colors showing the sample
density (low=yellow, high=red) and the limits for each input dimension.
The surface plot coloring SoC values (0=blue, 1=orange) at a given input
(I, U, T). The full dataset for the neural networks is based on this
approach.

### Outputs

Same time step: - Current (mapping on input)

Next time step for: - Terminal Voltage - Surface Temperature - State of
Charge - Open Circuit Voltage - Core Temperature

### Principle

![flow](./doc/flow.png) The data gets processed like in a standard
transformer model. We feed the input space as a time series into the
neural network, and it outputs the next time step. The output space
doesn’t include the current because this is randomly generated.
Predicting the current could be interesting for certain operation
strategies, but in our case, the current comes from an external source,
like a sensor in the real world. Voltage and surface temperature can
either come from the predicted time step in an autoregressive fashion or
also from an external source.

## Results (working on it)

At first, I thought we could skip a full hyperparameter optimization
(HPO) run and just tweak a few parameters of a well-known model like
GPT-2/3. But that didn’t work out. You can check out my initial attempt
[here](/doc/pred_plot.ipynb). I basically adjusted what I thought were
the key parameters and kept everything else close to the original
GPT-2/3 settings. Unfortunately, the results weren’t great. The model
learned the basic behavior, but as soon as it had to rely on its own
outputs (being autoregressive), the error rate shot up and became
unacceptable.

So, it’s clear that we need to do a full HPO run given the changes in
the dataset and the different input and output space compared to GPT-2/3
and my thesis requirements.

Results from hpo and transformer model training looking promising.
Planned uploading until mid August!

## References

<div id="refs" class="references csl-bib-body" entry-spacing="0">

<div id="ref-Sulzer2021" class="csl-entry">

<span class="csl-left-margin">\[1\]
</span><span class="csl-right-inline">V. Sulzer, S. G. Marquis, R.
Timms, M. Robinson, and S. J. Chapman, “Python Battery Mathematical
Modelling (PyBaMM),” *JORS*, vol. 9, no. 1, p. 14, Jun. 2021, doi:
[10.5334/jors.309](https://doi.org/10.5334/jors.309).</span>

</div>

<div id="ref-Touvron2023" class="csl-entry">

<span class="csl-left-margin">\[2\]
</span><span class="csl-right-inline">H. Touvron *et al.*, “LLaMA: Open
and Efficient Foundation Language Models.” arXiv, Feb. 27, 2023.
Accessed: May 05, 2024. \[Online\]. Available:
<http://arxiv.org/abs/2302.13971></span>

</div>

<div id="ref-LGChem2016" class="csl-entry">

<span class="csl-left-margin">\[3\]
</span><span class="csl-right-inline">LG Chem, “Datasheet INR21700 M50.”
Aug. 23, 2016. Available:
<https://www.dnkpower.com/wp-content/uploads/2019/02/LG-INR21700-M50-Datasheet.pdf></span>

</div>

</div>
