# Supplement: verification suite for annealed Fisher bounds, normalized-minor Ward identity, and one-contraction obstructions

This directory contains one standalone verification program, `replay.py`, for the
preprint "Annealed Fisher bounds, normalized-minor Ward identity, and obstructions to
one-contraction pairing ceilings in the Holstein model" (Davis Rens, SuperMatics Inc.,
2026).

The program verifies two distinct classes of statements:

1. **Exact rational and outward-interval checks (`CERTIFIED` within their finite scope):**
   - Exact rational spin-occupation identity $2(n_{\uparrow}-1/2)(n_{\downarrow}-1/2) = (n-1)^2 - 1/2$.
   - Exact rational integral $\int_0^1 u(1-u)(1-2u)^2\,du = 1/30$ used in the second-order obstruction.
   - 40-digit outward-interval enclosures (`mpmath.iv`) for $C_{\mathrm{vert}} = \zeta(3)/(16\pi^3)$,
     $C_{\mathrm{SE}} = 1/24 + \zeta(3)/(16\pi^3)$, $C_1^{\mathrm{phys}} = 1/24 + \zeta(3)/(8\pi^3) \in [0.0465126, 0.0465128]$,
     and $|(1-f)(1-3f)| \le 1$ on $f \in [0, 1]$ across 64 subintervals.

2. **Finite-lattice and finite-Fock-space checks (`NUMERICAL`):**
   - Exact 2-site dynamic conditional Fisher identity at $\Omega > 0$ via Kubo block-matrix exponential.
   - Grand-canonical vs. fixed-$N$ canonical matrix Fisher identities on a 2-site dimer, and
     a 3-site triangle negative-trace witness for $U_0 > 0$.
   - Normalized-minor Ward bound $\tau\|\partial_v Q_b\|_{\mathrm{HS}}^2 \le \frac{1}{8} v^T \chi_b v$
     across 50 random $L=6$ rings, plus the $\tau^{-1}$ growth of unnormalized minors.
   - Spectral one-contraction formula checked against direct Matsubara sums and 16-dimensional
     Fock-space time-ordered perturbation, plus pointwise Ward domination across 100 random $L=6$ rings.
   - Second-order non-Ward divergence on a $24\times 24$ 2D metal and a 3-site triangle, extensive
     coexistence covariance scaling on $L=4, 8, 12, 16$ rings, and a $20\times 20$ CDW domain wall.

The finite-lattice numerical checks are sanity checks only; they do not replace the analytical
proofs in the preprint, which hold for every finite graph and all parameters.

## Requirements

* `uv` 0.9.5 (or Python 3.11+ with `mpmath`, `numpy`, `scipy`)
* `mpmath`
* `numpy`
* `scipy`

## Commands

Default run:

```
uv run --no-project --with mpmath --with numpy --with scipy python replay.py
```

Guarded mutant run (corrupts one mathematical claim, sign, or scaling law at a time across
10 mutants and asserts every mutant fails with `AssertionError`):

```
uv run --no-project --with mpmath --with numpy --with scipy python replay.py --mutants
```

## Manifest

`MANIFEST.sha256` lists the SHA-256 digest of `replay.py` and `README.md`.
