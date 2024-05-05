# Cell-Li-Gent

I’m diving into a data-driven approach for multivariate time series
prediction, aiming to simplify the battery modeling process and cut down
on expensive battery tests. By fully harnessing the data, I’m using a
randomized protocol to create a synthetic dataset via electrochemical
simulation (DFN using [PyBaMM](https://github.com/pybamm-team/PyBaMM)).
These dataset then serves as fuel to train neural networks. Building on
what I laid out in my [thesis](/doc/thesis.pdf) proposal, the goal is to
not just predict battery behavior but to also uncover more efficient
ways to operate batteries using an intelligent agent.

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

## Realization

The [model](/model) directory has the designs I’m considering:

- Transformer based on Meta’s [Llama
  3](https://github.com/meta-llama/llama3) \[1\]

- Mamba (State-Space) based on
  [Mamba](https://github.com/mamba-org/mamba) \[2\]

- JEPA (Joint-Embeddings) based on Meta’s
  [I-JEPA](https://github.com/facebookresearch/ijepa) \[3\]

This project also heavily utilizes Andrej Karpathy’s
[nanoGPT](https://github.com/karpathy/nanoGPT/tree/master).

Why am I picking these architectures for battery state estimation? It’s
because Transformers are everywhere, the State-Space method has been
reliable in traditional battery modeling by considering historical
performance, and then there’s Jepa—it’s intriguing to see how battery
predictions fare when we manipulate data in a latent space.

Executing a full HPO/NAS would burn too many resources and is non
trivial considering the theoretically limitless dataset size. Hence,
I’ve used published hyperparameters and tailored them with the
[nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) scaling
suggestions for a single GPU setup with 80GB. The size for the generated
battery dataset is also based on the ratio of the tailored parameters
count to the original parameters count.

There’s a lot more interesting architectures and combinations
thereof—hopefully, as computational costs might distribute, digging into
these will be feasible.

## Result

… still brewing …

## References

<div id="refs" class="references csl-bib-body" entry-spacing="0">

<div id="ref-Touvron2023" class="csl-entry">

<span class="csl-left-margin">\[1\]
</span><span class="csl-right-inline">H. Touvron *et al.*, “LLaMA: Open
and Efficient Foundation Language Models.” arXiv, Feb. 27, 2023.
Accessed: May 05, 2024. \[Online\]. Available:
<http://arxiv.org/abs/2302.13971></span>

</div>

<div id="ref-Gu2023" class="csl-entry">

<span class="csl-left-margin">\[2\]
</span><span class="csl-right-inline">A. Gu and T. Dao, “Mamba:
Linear-Time Sequence Modeling with Selective State Spaces.” arXiv, Dec.
01, 2023. Accessed: Apr. 24, 2024. \[Online\]. Available:
<http://arxiv.org/abs/2312.00752></span>

</div>

<div id="ref-Assran2023" class="csl-entry">

<span class="csl-left-margin">\[3\]
</span><span class="csl-right-inline">M. Assran *et al.*,
“Self-Supervised Learning from Images with a Joint-Embedding Predictive
Architecture.” arXiv, Apr. 13, 2023. Accessed: Apr. 24, 2024.
\[Online\]. Available: <http://arxiv.org/abs/2301.08243></span>

</div>

</div>
