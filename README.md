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

- **T4 (Mechanism-deletion identifiability under hidden mechanisms).** `THEOREM_T4_T5.md`. The headline H1 result: under v1 + all-variables-observed, $\mathrm{do}(\neg m^\star)$ is **always identifiable** — for both observed and latent target mechanisms — via $P(V) / P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)) \cdot \prod P_0$. Surprising compared to Pearl: variable interventions can be unidentifiable due to hedges; mechanism interventions cannot, because they act at the level where the chain rule already factorizes.

- **T5 (Variable-intervention reduction).** `THEOREM_T4_T5.md`. Variable interventions on a HADMG reduce to standard Pearl ADMG identification on the bipartite blowup; the Shpitser-Pearl hedge criterion applies directly.

- **T6 (Hidden-variable extension under observed boundary).** `THEOREM_H1_PLUS.md`. T4's formula extends verbatim to HADMGs with hidden variables, provided $\partial m^\star = \mathrm{in}(m^\star) \cup \mathrm{out}(m^\star) \subseteq V^{\mathrm{obs}}$. Verified by `test_h1_plus.py`.

- **T7 (Boundary-violating reduction).** `THEOREM_H1_PLUS.md`. When the boundary contains hidden variables, identifiability reduces to a Pearl ADMG identification problem on the bipartite blowup, governed by Shpitser-Pearl 2006's hedge criterion. Stated as a reduction; the full hyper-hedge completeness theorem is the principal open conjecture.

## The asymmetry (T4 + T5)

| Intervention | Identifiability under hidden mechanisms |
|---|---|
| $\mathrm{do}(v = x)$ — variable | Pearl hedge criterion via $B^\dagger(\mathcal{M})$ — non-trivial, may fail |
| $\mathrm{do}(\neg m)$ — mechanism deletion | **Always identifiable** under v1 (T4) |
| $\mathrm{do}(m \to m')$ — mechanism replacement | **Always identifiable** under v1 (T4 + T3) |

Mechanism-level interventions are strictly easier to identify than Pearl variable interventions. This is the strongest formal argument for the framework's value.

## Open theorems

- **H1+ (full) — hyper-hedge completeness.** The reverse direction of T7's reduction: a hyper-hedge in $B^\dagger(\mathcal{M})$ obstructs identifiability. Lifting Shpitser-Pearl 2006's completeness proof through the bipartite blowup with deterministic-relations preservation. v1 has T6 (sufficient condition) + T7 (reduction) + the hyper-hedge definition; the completeness theorem itself is the natural research-grade extension.
- **H2 (mechanism-correlated noise).** Originally framed as "mechanism-correlated noise generates conditional independencies inexpressible by any finite-latent Pearl SCM." On reflection (see project history), likely false in distributional or interventional form — Pearl with sufficient latents matches any joint or interventional distribution. A salvageable reframing might be a complexity-theoretic separation (smallest matching Pearl SCM is exponential in $|V|$), but no candidate construction is in hand.
