# Cell-Li-Gent

A data-driven approach for multivariate time series prediction, aiming
to simplify the battery modeling process and cut down on expensive
battery tests. Using a randomized protocol to create a synthetic dataset
via electrochemical simulation (DFN from
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

## Overview (working on it!)

The [model](/model) directory has the designs I’m considering:

- Transformer based on Meta’s [Llama
  3](https://github.com/meta-llama/llama3) \[2\]

- Mamba (State-Space) based on
  [Mamba](https://github.com/state-spaces/mamba) \[3\]

- JEPA (Joint-Embeddings) based on Meta’s
  [I-JEPA](https://github.com/facebookresearch/ijepa) \[4\]

This project also heavily utilizes Andrej Karpathy’s
[nanoGPT](https://github.com/karpathy/nanoGPT/tree/master).

Why am I picking these architectures for battery state estimation? It’s
because Transformers are everywhere, the State-Space method has been
reliable in traditional battery modeling by considering historical
information, and then there’s Jepa—it’s intriguing to see how battery
predictions fare when we manipulate data in a latent space.

Executing a full HPO/NAS would burn too many resources and is non
trivial considering the theoretically limitless dataset size. Hence,
I’ll use published hyperparameters and adjust them for a single GPU
setup with 80GB.

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
\[5\]. For the seak of simplicity I am focusing on short-term current
(10s), continuous current, and critical current specific to SoC
ranges(SoC\<0.2, SoC\>0.8). A single profile guarantees one full charge
and discharge, with a zero current intervals for equilibrium
information. ![profiles3d](./doc/data_3d.png) The plane colors showing
the sample density \[low=yellow, high=red\] and the limits for each
input dimension. The surface plot coloring SoC values \[0=blue,
1=orange\] at a given input \[I, U, T\]. The full dataset for the neural
networks is based on this approach.

### Outputs

Making model only current dependent: (…success not guaranteed)

- Terminal Voltage
- Surface Temperature

Getting additional information:

- State of Charge
- Open Circuit Voltage
- Battery Internal Temperature

## Results (…still brewing…)

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

<div id="ref-Gu2023" class="csl-entry">

<span class="csl-left-margin">\[3\]
</span><span class="csl-right-inline">A. Gu and T. Dao, “Mamba:
Linear-Time Sequence Modeling with Selective State Spaces.” arXiv, Dec.
01, 2023. Accessed: Apr. 24, 2024. \[Online\]. Available:
<http://arxiv.org/abs/2312.00752></span>

</div>

<div id="ref-Assran2023" class="csl-entry">

<span class="csl-left-margin">\[4\]
</span><span class="csl-right-inline">M. Assran *et al.*,
“Self-Supervised Learning from Images with a Joint-Embedding Predictive
Architecture.” arXiv, Apr. 13, 2023. Accessed: Apr. 24, 2024.
\[Online\]. Available: <http://arxiv.org/abs/2301.08243></span>

</div>

<div id="ref-LGChem2016" class="csl-entry">

<span class="csl-left-margin">\[5\]
</span><span class="csl-right-inline">LG Chem, “Datasheet INR21700 M50.”
Aug. 23, 2016. Available:
<https://www.dnkpower.com/wp-content/uploads/2019/02/LG-INR21700-M50-Datasheet.pdf></span>

</div>

</div>
