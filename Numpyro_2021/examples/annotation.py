# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Models of Annotation
======================================

In this example, we run MCMC for various crowdsourced annotation models in [1].

All models have discrete latent variables. Under the hood, we enumerate over
(marginalize out) those discrete latent sites in inference. Those models have different
complexity so they are great refererences for those who are new to Pyro/NumPyro
enumeration mechanism. We recommend readers compare the implementations with the
corresponding plate diagrams in [1] to see how concise a Pyro/NumPyro program is.

The interested readers can also refer to [3] for more explanation about enumeration.

The data is taken from Table 1 of reference [2].

Currently, this example does not include postprocessing steps to deal with "Label
Switching" issue (mentioned in section 6.2 of [1]).

**References:**

    1. Paun, S., Carpenter, B., Chamberlain, J., Hovy, D., Kruschwitz, U.,
       and Poesio, M. (2018). "Comparing bayesian models of annotation"
       (https://www.aclweb.org/anthology/Q18-1040/)
    2. Dawid, A. P., and Skene, A. M. (1979).
       "Maximum likelihood estimation of observer error‐rates using the EM algorithm"
    3. "Inference with Discrete Latent Variables"
       (http://pyro.ai/examples/enumeration.html)

"""

import argparse
import os

import numpy as np

from jax import nn, random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam


def get_data():
    """
    :return: a tuple of annotator indices and class indices. The first term has shape
        `num_positions` whose entries take values from `0` to `num_annotators - 1`.
        The second term has shape `num_items x num_positions` whose entries take values
        from `0` to `num_classes - 1`.
    """
    # NB: the first annotator assessed each item 3 times
    positions = np.array([1, 1, 1, 2, 3, 4, 5])
    annotations = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [3, 3, 3, 4, 3, 3, 4],
            [1, 1, 2, 2, 1, 2, 2],
            [2, 2, 2, 3, 1, 2, 1],
            [2, 2, 2, 3, 2, 2, 2],
            [2, 2, 2, 3, 3, 2, 2],
            [1, 2, 2, 2, 1, 1, 1],
            [3, 3, 3, 3, 4, 3, 3],
            [2, 2, 2, 2, 2, 2, 3],
            [2, 3, 2, 2, 2, 2, 3],
            [4, 4, 4, 4, 4, 4, 4],
            [2, 2, 2, 3, 3, 4, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 3, 2, 1, 2],
            [1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 1],
            [2, 2, 2, 1, 3, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 1],
            [2, 2, 2, 3, 2, 2, 2],
            [2, 2, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 3, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 1, 1, 2, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 2, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 3, 2, 3, 2],
            [4, 3, 3, 4, 3, 4, 3],
            [2, 2, 1, 2, 2, 3, 2],
            [2, 3, 2, 3, 2, 3, 3],
            [3, 3, 3, 3, 4, 3, 2],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 2, 1, 1, 1],
            [2, 3, 2, 2, 2, 2, 2],
            [1, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2],
        ]
    )
    # we minus 1 because in Python, the first index is 0
    return positions - 1, annotations - 1


def multinomial(annotations):
    """
    This model corresponds to the plate diagram in Figure 1 of reference [1].
    """
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample("zeta", dist.Dirichlet(jnp.ones(num_classes)))

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        with numpyro.plate("position", num_positions):
            numpyro.sample("y", dist.Categorical(zeta[c]), obs=annotations)


def dawid_skene(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 2 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            beta = numpyro.sample("beta", dist.Dirichlet(jnp.ones(num_classes)))

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        # here we use Vindex to allow broadcasting for the second index `c`
        # ref: http://num.pyro.ai/en/latest/utilities.html#numpyro.contrib.indexing.vindex
        with numpyro.plate("position", num_positions):
            numpyro.sample(
                "y", dist.Categorical(Vindex(beta)[positions, c, :]), obs=annotations
            )


def mace(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 3 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators):
        epsilon = numpyro.sample("epsilon", dist.Dirichlet(jnp.full(num_classes, 10)))
        theta = numpyro.sample("theta", dist.Beta(0.5, 0.5))

    with numpyro.plate("item", num_items, dim=-2):
        # NB: using constant logits for discrete uniform prior
        # (NumPyro does not have DiscreteUniform distribution yet)
        c = numpyro.sample("c", dist.Categorical(logits=jnp.zeros(num_classes)))

        with numpyro.plate("position", num_positions):
            s = numpyro.sample("s", dist.Bernoulli(1 - theta[positions]))
            probs = jnp.where(
                s[..., None] == 0, nn.one_hot(c, num_classes), epsilon[positions]
            )
            numpyro.sample("y", dist.Categorical(probs), obs=annotations)


def hierarchical_dawid_skene(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 4 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        # NB: we define `beta` as the `logits` of `y` likelihood; but `logits` is
        # invariant up to a constant, so we'll follow [1]: fix the last term of `beta`
        # to 0 and only define hyperpriors for the first `num_classes - 1` terms.
        zeta = numpyro.sample(
            "zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        omega = numpyro.sample(
            "Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            # non-centered parameterization
            with handlers.reparam(config={"beta": LocScaleReparam(0)}):
                beta = numpyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
            # pad 0 to the last item
            beta = jnp.pad(beta, [(0, 0)] * (jnp.ndim(beta) - 1) + [(0, 1)])

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        with numpyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :]
            numpyro.sample("y", dist.Categorical(logits=logits), obs=annotations)


def item_difficulty(annotations):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        eta = numpyro.sample(
            "eta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        chi = numpyro.sample(
            "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        with handlers.reparam(config={"theta": LocScaleReparam(0)}):
            theta = numpyro.sample("theta", dist.Normal(eta[c], chi[c]).to_event(1))
            theta = jnp.pad(theta, [(0, 0)] * (jnp.ndim(theta) - 1) + [(0, 1)])

        with numpyro.plate("position", annotations.shape[-1]):
            numpyro.sample("y", dist.Categorical(logits=theta), obs=annotations)


def logistic_random_effects(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample(
            "zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        omega = numpyro.sample(
            "Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )
        chi = numpyro.sample(
            "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            with handlers.reparam(config={"beta": LocScaleReparam(0)}):
                beta = numpyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
                beta = jnp.pad(beta, [(0, 0)] * (jnp.ndim(beta) - 1) + [(0, 1)])

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        with handlers.reparam(config={"theta": LocScaleReparam(0)}):
            theta = numpyro.sample("theta", dist.Normal(0, chi[c]).to_event(1))
            theta = jnp.pad(theta, [(0, 0)] * (jnp.ndim(theta) - 1) + [(0, 1)])

        with numpyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :] - theta
            numpyro.sample("y", dist.Categorical(logits=logits), obs=annotations)


NAME_TO_MODEL = {
    "mn": multinomial,
    "ds": dawid_skene,
    "mace": mace,
    "hds": hierarchical_dawid_skene,
    "id": item_difficulty,
    "lre": logistic_random_effects,
}


def main(args):
    annotators, annotations = get_data()
    model = NAME_TO_MODEL[args.model]
    data = (
        (annotations,)
        if model in [multinomial, item_difficulty]
        else (annotators, annotations)
    )

    mcmc = MCMC(
        NUTS(model),
        args.num_warmup,
        args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(random.PRNGKey(0), *data)
    mcmc.print_summary()


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.6.0")
    parser = argparse.ArgumentParser(description="Bayesian Models of Annotation")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument(
        "--model",
        nargs="?",
        default="ds",
        help='one of "mn" (multinomial), "ds" (dawid_skene), "mace",'
        ' "hds" (hierarchical_dawid_skene),'
        ' "id" (item_difficulty), "lre" (logistic_random_effects)',
    )
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
'''
                 mean       std    median      5.0%     95.0%     n_eff     r_hat
beta[0,0,0]      0.84      0.05      0.84      0.78      0.93   2100.47      1.00
beta[0,0,1]      0.13      0.04      0.12      0.06      0.19   1730.60      1.00
beta[0,0,2]      0.02      0.02      0.01      0.00      0.04   1198.02      1.00
beta[0,0,3]      0.02      0.02      0.01      0.00      0.04   1341.98      1.00
beta[0,1,0]      0.11      0.14      0.06      0.00      0.28    269.73      1.02
beta[0,1,1]      0.12      0.14      0.08      0.00      0.30    151.86      1.02
beta[0,1,2]      0.48      0.30      0.52      0.01      0.86     16.31      1.13
beta[0,1,3]      0.29      0.24      0.21      0.00      0.66     28.69      1.08
beta[0,2,0]      0.06      0.03      0.05      0.01      0.11   1409.52      1.00
beta[0,2,1]      0.85      0.05      0.85      0.77      0.92   1683.85      1.00
beta[0,2,2]      0.08      0.03      0.07      0.02      0.13   2160.94      1.00
beta[0,2,3]      0.02      0.02      0.01      0.00      0.04   1335.51      1.00
beta[0,3,0]      0.10      0.11      0.06      0.00      0.24    184.88      1.01
beta[0,3,1]      0.11      0.12      0.07      0.00      0.26    377.35      1.01
beta[0,3,2]      0.51      0.30      0.60      0.00      0.85     25.04      1.11
beta[0,3,3]      0.28      0.23      0.21      0.00      0.65     30.13      1.10
beta[1,0,0]      0.71      0.10      0.72      0.55      0.87   2023.54      1.00
beta[1,0,1]      0.20      0.09      0.19      0.05      0.33   2237.43      1.00
beta[1,0,2]      0.04      0.04      0.03      0.00      0.10   1351.09      1.00
beta[1,0,3]      0.04      0.04      0.03      0.00      0.11   1394.38      1.00
beta[1,1,0]      0.16      0.15      0.12      0.00      0.39    287.86      1.02
beta[1,1,1]      0.16      0.15      0.11      0.00      0.34    474.15      1.01
beta[1,1,2]      0.33      0.22      0.30      0.00      0.63     33.88      1.06
beta[1,1,3]      0.35      0.20      0.33      0.02      0.64    234.90      1.01
beta[1,2,0]      0.08      0.05      0.07      0.01      0.16   1839.55      1.00
beta[1,2,1]      0.52      0.10      0.52      0.36      0.69   2102.83      1.00
beta[1,2,2]      0.36      0.10      0.35      0.21      0.53   1820.43      1.00
beta[1,2,3]      0.04      0.04      0.03      0.00      0.10   1315.40      1.00
beta[1,3,0]      0.15      0.14      0.11      0.00      0.37    462.80      1.01
beta[1,3,1]      0.15      0.14      0.11      0.00      0.35    294.70      1.01
beta[1,3,2]      0.34      0.21      0.33      0.00      0.63     54.81      1.05
beta[1,3,3]      0.36      0.19      0.34      0.04      0.65    442.92      1.01
beta[2,0,0]      0.87      0.07      0.88      0.76      0.98   1537.63      1.00
beta[2,0,1]      0.04      0.04      0.03      0.00      0.10   1161.34      1.00
beta[2,0,2]      0.04      0.04      0.03      0.00      0.09   1173.04      1.00
beta[2,0,3]      0.05      0.05      0.03      0.00      0.10    929.25      1.00
beta[2,1,0]      0.16      0.16      0.11      0.00      0.40    502.10      1.01
beta[2,1,1]      0.22      0.16      0.18      0.00      0.47   1471.88      1.00
beta[2,1,2]      0.27      0.18      0.24      0.00      0.53    920.81      1.01
beta[2,1,3]      0.34      0.18      0.33      0.06      0.64   1118.96      1.00
beta[2,2,0]      0.10      0.07      0.09      0.01      0.19   1264.60      1.00
beta[2,2,1]      0.70      0.09      0.70      0.55      0.83   1043.25      1.00
beta[2,2,2]      0.16      0.08      0.15      0.04      0.28   1367.91      1.00
beta[2,2,3]      0.04      0.04      0.03      0.00      0.09   1400.98      1.00
beta[2,3,0]      0.15      0.15      0.11      0.00      0.36    323.68      1.02
beta[2,3,1]      0.22      0.16      0.19      0.00      0.44    828.61      1.00
beta[2,3,2]      0.28      0.17      0.26      0.00      0.52    701.91      1.00
beta[2,3,3]      0.34      0.17      0.33      0.08      0.61   1386.25      1.00
beta[3,0,0]      0.80      0.08      0.81      0.68      0.93   1811.71      1.00
beta[3,0,1]      0.11      0.06      0.10      0.02      0.20   1391.92      1.00
beta[3,0,2]      0.05      0.04      0.03      0.00      0.10   1116.63      1.00
beta[3,0,3]      0.04      0.04      0.03      0.00      0.10    928.74      1.00
beta[3,1,0]      0.16      0.14      0.12      0.00      0.35    452.21      1.02
beta[3,1,1]      0.16      0.14      0.11      0.00      0.37    218.39      1.01
beta[3,1,2]      0.38      0.24      0.37      0.01      0.70     27.29      1.09
beta[3,1,3]      0.31      0.20      0.28      0.00      0.59    102.15      1.03
beta[3,2,0]      0.08      0.05      0.07      0.00      0.15   2490.72      1.00
beta[3,2,1]      0.67      0.09      0.68      0.55      0.83   1784.51      1.00
beta[3,2,2]      0.16      0.07      0.15      0.04      0.26   1869.00      1.00
beta[3,2,3]      0.08      0.06      0.07      0.00      0.16   1432.38      1.00
beta[3,3,0]      0.15      0.14      0.10      0.00      0.34    505.62      1.01
beta[3,3,1]      0.15      0.14      0.10      0.00      0.35    221.37      1.01
beta[3,3,2]      0.40      0.24      0.41      0.01      0.73     34.63      1.08
beta[3,3,3]      0.30      0.20      0.26      0.00      0.59    123.29      1.03
beta[4,0,0]      0.84      0.08      0.86      0.73      0.96   1751.22      1.00
beta[4,0,1]      0.07      0.05      0.05      0.00      0.14   1219.57      1.00
beta[4,0,2]      0.05      0.05      0.03      0.00      0.11   1012.88      1.00
beta[4,0,3]      0.04      0.04      0.03      0.00      0.10   1105.88      1.00
beta[4,1,0]      0.16      0.15      0.11      0.00      0.37    358.37      1.00
beta[4,1,1]      0.21      0.16      0.18      0.00      0.42   1259.07      1.00
beta[4,1,2]      0.33      0.19      0.32      0.00      0.60     96.01      1.03
beta[4,1,3]      0.30      0.20      0.26      0.01      0.58    296.74      1.02
beta[4,2,0]      0.17      0.08      0.16      0.04      0.28   1664.12      1.00
beta[4,2,1]      0.59      0.10      0.59      0.46      0.78   1641.63      1.00
beta[4,2,2]      0.20      0.08      0.19      0.06      0.33   1694.68      1.00
beta[4,2,3]      0.04      0.04      0.03      0.00      0.09   1565.56      1.00
beta[4,3,0]      0.15      0.15      0.10      0.00      0.36    388.88      1.01
beta[4,3,1]      0.21      0.15      0.18      0.01      0.42   1564.77      1.00
beta[4,3,2]      0.35      0.20      0.35      0.00      0.62     92.71      1.02
beta[4,3,3]      0.30      0.19      0.27      0.01      0.57    242.71      1.02
      pi[0]      0.40      0.07      0.40      0.30      0.51   2035.98      1.00
      pi[1]      0.08      0.05      0.07      0.00      0.15     37.18      1.08
      pi[2]      0.43      0.07      0.43      0.31      0.54   1658.41      1.00
      pi[3]      0.09      0.06      0.08      0.00      0.17     44.84      1.05

Number of divergences: 0
'''