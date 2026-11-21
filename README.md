# Annealed Fisher bounds, normalized-minor Ward identity, and obstructions to one-contraction pairing ceilings in the Holstein model

Davis Rens, SuperMatics Inc. ([ORCID 0000-0001-5839-5197](https://orcid.org/0000-0001-5839-5197)).
Version 1.0.0.

Paper DOI: <https://doi.org/10.5281/zenodo.24413070>
Replay supplement DOI: <https://doi.org/10.5281/zenodo.24413072>
Source: <https://github.com/SuperMatics-Inc/holstein-annealed-fisher-bounds>

## What this deposit is

This repository carries the preprint `paper/main.pdf` (13 pages) and the complete replay supplement for
it. The paper is an intermediate, finite-volume methodological result. It is not a pairing ceiling: it
proves no frequency-linear bound on the singlet pair-decay temperature of the Holstein model, and it does
not close that open step. The frequency-linear upper bound `k_B T_pair <= C(t,U) Omega` of the spinful
Holstein model, uniformly in volume and chemical potential as `Omega -> 0`, remains open, and the paper
says so in its abstract.

What the paper does establish, for every finite graph and all parameters:

1. An annealed Fisher-information inequality. Integration by parts on the static-mode marginal and on the
   exact interacting finite-`Omega` conditional marginal, for any Hubbard coupling `U_0 <= 0`, gives the
   full `V x V` matrix bound `E[chi_eff] <= U^-1 I_V`, with no small-hopping cutoff, no graph ball-growth
   hypothesis and no pointwise convexity assumption. In the grand-canonical ensemble at `U_0 = 0` this
   controls the expected near-Fermi spectral weight by `2 cosh^2(a/2) tau/U`; in fixed-particle-number
   sectors the uniform-shift compressibility vanishes identically while local diagonal entries stay
   bounded by `U^-1`.
2. A normalized-minor Ward identity. A Hilbert-Schmidt weighted Fisher inequality combined with a
   Daletskii-Krein spectral identity for the normalized pair-minor matrix `Q_b = (I + exp(-beta h_b))^-1`
   gives `tau ||d_v Q_b||_HS^2 <= (1/8) v^T chi_b v` uniformly in `b, v, tau, t, mu`, which removes the
   `tau^-1` derivative loss carried by unnormalized minors.
3. An exact annealed one-contraction bound, `E_0[|dS^(1)(b_0)|] <= C1_phys (Omega/tau)^2`, with
   `C1_phys = 1/24 + zeta(3)/(8 pi^3) = 0.0465127...`, enclosed as `[0.0465126, 0.0465128]`, and no
   logarithmic factors.

It also exhibits three sharp obstructions that block extending that one-contraction bound into a closed
global inequality for the interacting structure factor `S_Omega`: a `Theta(tau^-2)` second-derivative
blowup in clean two-dimensional metals, an extensive `Theta(V (Omega/tau)^2)` coexistence covariance under
static-to-dynamic measure reweighting, and the mismatch between conditional and bare static
compressibilities together with the `(t,U)`-blindness of a single bare contraction. Metallic
charge-density-wave domain walls near coexistence are assessed and shown to defeat unresummed annealed
perturbation without providing a physical counterexample to a linear ceiling.

## Contents

```
MANIFEST.sha256          SHA-256 of every other file in this tree
CITATION.cff             citation metadata for the preprint and this supplement
LICENSE                  Apache License 2.0
paper/main.pdf           the reviewed manuscript, released unmodified
paper/main.tex           LaTeX source of that manuscript
paper/refs.bib           bibliography
replay/replay.py         the standalone verification program
replay/README.md         what the program verifies and how to run it
replay/MANIFEST.sha256   SHA-256 of replay.py and replay/README.md, pinned since review
```

`paper/main.pdf` is the reviewed artifact and is released byte for byte; its SHA-256 digest is
`0845717f4f10349b7d592e1d79f5a78a589ea386a987433863c41cd1630c289c`. Continuous integration verifies
those bytes, the text they carry and the replay program; it does not rebuild the PDF from `main.tex`, so
the deposit is exactly the reviewed document.

## Replay

`replay/replay.py` needs `mpmath`, `numpy` and `scipy`; no data file and no network access are used.

```
python replay/replay.py            # every check must pass, exit code 0
python replay/replay.py --mutants  # ten corrupted variants must each fail, exit code 0
```

The default run prints one `[CERTIFIED]` line and four `[NUMERICAL]` lines and ends with
`ALL VERIFICATION CHECKS PASSED.`; it covers exact rational identities, 40-digit `mpmath.iv` outward
interval enclosures of `C_vert = zeta(3)/(16 pi^3)`, `C_SE = 1/24 + zeta(3)/(16 pi^3)` and
`C1_phys = 1/24 + zeta(3)/(8 pi^3)`, the 2-site dynamic conditional Fisher identity through a Kubo
block-matrix exponential, grand-canonical versus fixed-`N` canonical matrix Fisher identities on a
2-site dimer, a 3-site negative-trace witness for `U_0 > 0`, the normalized-minor Ward bound across 50
random `L = 6` rings together with the `tau^-1` growth of unnormalized minors, the spectral
one-contraction formula against direct Matsubara sums and 16-dimensional Fock-space time-ordered
perturbation with pointwise Ward domination across 100 random `L = 6` rings, and the numerical
obstruction witnesses on a `24 x 24` two-dimensional metal, on `L = 4, 8, 12, 16` rings and on a
`20 x 20` charge-density-wave domain wall. The mutant run ends with `ALL 10 MUTANTS REJECTED.`

## Status and disclosures

The finite-lattice, sixteen-dimensional Fock-space and quadrature diagnostics in the supplement are
numerical sanity checks only and do not substitute for the analytical proofs, which hold for every
finite graph. No superconducting transition is claimed and no pairing ceiling is established.

AI agents generated the proofs, code and drafts under D. Rens' direction, and separate AI agents carried
out the independent reviews of the cited records. This preprint has not been peer reviewed.
