# MH-within-Gibbs building blocks (Gustav Robertson)

This sub-package contains the sampler building blocks written by Gustav
Robertson (hierarchical MH-within-Gibbs algorithm, Robertson et al.,
2025). They are used by `../runner.py`.

| File | Content |
|---|---|
| `states.py` | `state(mu, sig, X)` |
| `proposals.py` | Gaussian random-walk proposals for `mu_d`, `sig_d`, `X[e]` |
| `custom_logps.py` | log-densities of the blocks (`logp_mu`, `logp_sig`, `logp_X`, `logp_hyper`) |
| `random_walk.py` | `mh_sampler` (one Metropolis-Hastings block) |
| `adaptation.py` | proposal adaptation: acceptance-rate scaling and covariance adaptation (Haario et al., 2001) |
| `conjugate_samplers.py` | conjugate updates for `mu` and `sig` (not used by default) |

Changes with respect to the original files are listed in the header of
each file. They are limited to:

1. package-relative imports (`from .states import state` instead of
   `from lib.states import state`);
2. removal of superseded classes (`*_old`) and of commented-out code;
3. comments translated to English.

The algorithm itself is unchanged.
