# Cell-Li-Gent

A data-driven approach for multivariate time series prediction, aiming
to simplify the battery modeling process and cut down on expensive
battery tests. By fully harnessing the data, I’m using a randomized
protocol to create a synthetic dataset via electrochemical simulation
(DFN using [PyBaMM](https://github.com/pybamm-team/PyBaMM) \[1\]). This
dataset then serves as fuel to train neural networks. Building on what I
laid out in my [thesis](/doc/thesis.pdf) proposal, the goal is to not
just predict battery behavior but to also uncover more efficient ways to
operate batteries using an intelligent agent.

Here’s a snapshot from my [thesis](/doc/thesis.pdf) where I compared a
classic method against a neural network that doesn’t go overboard on the
parameters (~350k) but still delivers performance that’s pretty much on
par with the traditional model:

| Dataset | Battery-Model             | MSE    | MAE    |
|---------|---------------------------|--------|--------|
| LFP     | Data-Driven (Transformer) | 7.3e-4 | 2.0e-2 |
| LFP     | Classical (ECM)           | 3.5e-4 | 1.5e-2 |
| NMC     | Data-Driven (Transformer) | 8.3e-5 | 8.0e-3 |
| NMC     | Classical (ECM)           | 1.0e-7 | 2.9e-4 |

## Overview (…working on it!)

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

Snapshot of the generated profiles, which come in three different
styles. The first one is a step profile (blue). The second (green)
combines charging and discharging phases from a drive cycle and tailored
to the cell. The third one (red) is a superimposed version of the both,
smoothed out with a Gaussian kernel. These profiles are supposed to
cover static and dynamic aspects. ![profiles](./doc/profiles_plot.png)
Throughout the entire sequence, the profiles are designed to stay within
the SoC ranges and current values specified in the datasheet \[5\]. To
keep it simple, we’re focusing on a few key areas: short-term current,
continuous current, and critical current specific to SoC ranges. The
profile guarantees a complete charge and discharge, with zero current
periods for relaxation. ![profiles3d](./doc/data_3d.png) All these
profiles are based on a random generated current profile with datasheet
based constraints. The 3d illustration shows where the highest sample
density is and also what soc is present at a given point. The full
training dataset for the AI models is based on this approach. There are
almost limitless options to refine the data generation but this might be
addressed after a first evaluation.

Inputs for the training are:

- Terminal Current
- Terminal Voltage
- X-Avg. Temperature

Prediction parameters are:

- State of Charge
- tbd…

## Result (…still brewing…)

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
