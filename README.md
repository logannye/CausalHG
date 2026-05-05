# Causal Hypergraphs

A working repository for the formal development of **Causal Hypergraph SCMs** — a strict generalization of Pearl's structural causal models in which mechanisms (hyperedges) are first-class causal objects, intervenable in their own right.

## Why

Pearl's SCM framework rests on three structural assumptions that are easy to miss:

1. Each structural equation produces a single output variable.
2. Causal edges are dyadic.
3. The induced graph is a DAG over variables.

Many real domains — chemical reaction networks, regulatory pathways, n-ary knowledge tuples, multi-target therapies, joint policy interventions — have **genuinely n-ary mechanisms with multiple co-produced outputs and joint constraints (conservation, stoichiometry, irreducible synergy)**. These survive intact only if the mechanism itself is named in the formalism. The hypergraph framework provides this naming.

The contribution is not new *expressive power* (Pearl with latents is universal) but **first-class addressability** of mechanisms. Once mechanisms are named, the do-vocabulary strictly extends — `do(¬m)` (delete a mechanism) and `do(m → m')` (replace a mechanism) are well-defined operations with no Pearl analogue as single interventions.

## Project structure

```
.
├── README.md                  this file
├── FOUNDATIONS.md             v1 formal definitions, axioms, conventions
├── MINIMAL_EXAMPLE.md         worked 2-mechanism reaction network with full derivations
├── THEOREM_T1.md              soundness + completeness proof of bipartite-blowup d*-separation
├── THEOREM_T2_T3.md           identifiability theorems for mechanism deletion + replacement
├── THEOREM_T4_T5.md           HADMG identifiability: T4 (mechanism-deletion robust to latents) + T5
├── THEOREM_H1_PLUS.md         T6 (hidden-variable extension) + T7 (Pearl-ID reduction) + hyper-hedge
├── whitepaper.md              section-skeleton, populated as theory stabilizes
└── minimal_model/
    ├── __init__.py
    ├── scm.py                 HypergraphSCM, Mechanism (with latent flag), Factor, validate(),
    │                          factorize(), truncated_factorization(),
    │                          is_mechanism_deletion_identifiable(), three do-operators, abduction
    ├── examples.py            reaction_network(), reaction_network_with_latent_BE(),
    │                          reaction_network_with_hidden_W()
    ├── dseparation.py         d*-separation algorithm with deterministic-relations augmentation
    ├── test_example.py        verifies hand calculations from MINIMAL_EXAMPLE.md
    ├── test_dseparation.py    verifies T1 predictions
    ├── test_factorization.py  verifies T2/T3
    ├── test_hadmg.py          verifies T4 under latent-mechanism confounding
    └── test_h1_plus.py        verifies T6/T7 with a hidden variable W
```

## v1 conventions (committed)

- **C1.** Mechanism-level acyclicity (the hyperedge dependency graph is a DAG).
- **C2.** Deterministic structural functions with explicit per-mechanism noise (Markov-kernel generalization deferred).
- **C3.** Bipartite role typing — each hyperedge has an `(in, out)` partition only (richer roles deferred).
- **C4.** Single producer — each variable has at most one producing mechanism (required for the mechanism-level chain rule, T2).

These keep v1 tractable. Alternatives are catalogued in `FOUNDATIONS.md` §11 as future work.

## Scope of v1

Tight defensible core, not a full theory:

1. Formal definition of Hypergraph SCM with typed incidence and joint structural functions.
2. The three-flavor do-operator.
3. **One** target theorem: bipartite-blowup d-separation is sound and complete.
4. **One** worked example exhibiting irreducibility to Pearl.
5. Honest open-problems section: full do-calculus extension, mechanism-level identifiability, cyclic case, hyper-confounding.

## How to run the model

```bash
cd "Causal Hypergraph Project"
python -m minimal_model.test_example         # 11 tests: structural, sampling, interventions, counterfactuals
python -m minimal_model.test_dseparation     # 12 tests: T1 predictions, including det-augmentation
python -m minimal_model.test_factorization   # 11 tests: validation, T2 truncation, T3 replacement
python -m minimal_model.test_hadmg           # 16 tests: HADMG, T4 under latent confounding
python -m minimal_model.test_h1_plus         # 10 tests: T6/T7 with a hidden variable W
```

No dependencies beyond NumPy. **60/60 tests passing.**

## Theorems proved

- **T1 (Bipartite-blowup d*-separation, soundness + completeness).** `THEOREM_T1.md`. Reduces hypergraph d-separation to standard Pearl d-separation on the noise-augmented blowup, with the deterministic-relations augmentation (Geiger-Pearl 1990) handling joint outputs.

- **Lemma 1.1 (Mechanism-level chain rule), T2 (mechanism-deletion truncated factorization), T3 (mechanism replacement).** `THEOREM_T2_T3.md`. Under causal sufficiency and v1 conventions C1–C4, mechanism-level interventions are identifiable.

- **T4 (Mechanism-deletion identifiability under hidden mechanisms).** `THEOREM_T4_T5.md`. Under v1 + all-variables-observed, $\mathrm{do}(\neg m^\star)$ is identifiable in closed form — for both observed and latent target mechanisms — via $P(V) / P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)) \cdot \prod P_0$. The formula is uniform in $m^\star$'s observability and requires no Pearl-style hedge search; it reads off the chain-rule factorization directly.

- **T5 (Variable-intervention reduction).** `THEOREM_T4_T5.md`. Variable interventions on a HADMG reduce to standard Pearl ADMG identification on the bipartite blowup; the Shpitser-Pearl hedge criterion applies directly.

- **T6 (Hidden-variable extension under observed boundary).** `THEOREM_H1_PLUS.md`. T4's formula extends verbatim to HADMGs with hidden variables, provided $\partial m^\star = \mathrm{in}(m^\star) \cup \mathrm{out}(m^\star) \subseteq V^{\mathrm{obs}}$. Verified by `test_h1_plus.py`.

- **T7 (Boundary-violating reduction).** `THEOREM_H1_PLUS.md`. When the boundary contains hidden variables, identifiability reduces to a Pearl ADMG identification problem on the bipartite blowup, governed by Shpitser-Pearl 2006's hedge criterion. Stated as a reduction; the full hyper-hedge completeness theorem is the principal open conjecture.

## The asymmetry (T4 + T5)

| Intervention | Identifying expression under v1 + $V^{\mathrm{lat}} = \emptyset$ |
|---|---|
| $\mathrm{do}(\neg m)$ — mechanism deletion | **Closed form (T4):** $P(V) / P(\mathrm{out}(m) \mid \mathrm{in}(m)) \cdot \prod P_0$. Always identifiable; $O(\lvert V \rvert + \lvert E \rvert)$. |
| $\mathrm{do}(m \to m')$ — mechanism replacement | **Closed form (T3 + T4):** same denominator, replacement factor in numerator. |
| $\mathrm{do}(v = x)$ — variable intervention | **Reduces (T5)** to Pearl multi-variable ID on $B^\dagger(\mathcal{M})$ — Pearl machinery applies verbatim. |

The asymmetry has two components: (i) **closed-form vs case-analytic** — mechanism interventions admit a single identifying formula readable from the chain rule, while variable interventions require running Pearl's ID algorithm on the bipartite blowup; (ii) **robustness to hidden mechanisms** — T4's formula needs only the typed incidence of $m^\star$, regardless of whether $m^\star$ is observed or latent. Under hidden *variables* (`THEOREM_H1_PLUS.md`), the asymmetry sharpens: T6 closes the observed-boundary case in closed form, while T7's boundary-violating case can genuinely fail via a hyper-hedge.

The framework's value proposition is therefore not new identifiability where Pearl's machinery fails, but (i) a closed-form identifier requiring no search, (ii) an intervention vocabulary that matches experimental practice (one drug ablation = one $\mathrm{do}(\neg m)$, not a multi-variable Pearl translation), and (iii) under hidden variables, a clean sufficient condition that bypasses hyper-hedge analysis.

## Open theorems

- **H1+ (full) — hyper-hedge completeness.** The reverse direction of T7's reduction: a hyper-hedge in $B^\dagger(\mathcal{M})$ obstructs identifiability. Lifting Shpitser-Pearl 2006's completeness proof through the bipartite blowup with deterministic-relations preservation. v1 has T6 (sufficient condition) + T7 (reduction) + the hyper-hedge definition; the completeness theorem itself is the natural research-grade extension.
- **H2 (mechanism-correlated noise).** Originally framed as "mechanism-correlated noise generates conditional independencies inexpressible by any finite-latent Pearl SCM." On reflection (see project history), likely false in distributional or interventional form — Pearl with sufficient latents matches any joint or interventional distribution. A salvageable reframing might be a complexity-theoretic separation (smallest matching Pearl SCM is exponential in $|V|$), but no candidate construction is in hand.
