# Theorems T4 and T5: Identifiability Under Hidden Mechanisms

This document is the principal H1 contribution. It addresses identifiability of post-intervention distributions in **Hypergraph ADMGs (HADMGs)** — the analogue of Pearl's acyclic directed mixed graphs in the hypergraph framework.

The headline finding is *not* that the hypergraph framework needs its own version of Shpitser-Pearl 2006's hedge criterion — it is that **mechanism-deletion identifiability admits a single closed-form identifier** read directly from the chain rule (no Pearl ID search), uniformly in whether the target mechanism is observed or latent. Variable interventions reduce verbatim to Pearl multi-variable ID on the bipartite blowup (T5). The asymmetry — closed-form vs case-analytic — is structural: mechanism-level interventions act at the level where the chain rule already factorizes. The asymmetry sharpens further under hidden variables (T6/T7 in `THEOREM_H1_PLUS.md`).

---

## 0. Hypergraph ADMG (HADMG)

A **Hypergraph ADMG** is a Hypergraph SCM in which a subset $E^{\mathrm{lat}} \subseteq E$ of mechanisms is marked as **latent** — their structural functions $f_m$ are unknown, but their typed incidence $\rho(m)$ is known. Observed mechanisms $E^{\mathrm{obs}} = E \setminus E^{\mathrm{lat}}$ have known structural functions.

For v1, we restrict attention to HADMGs in which every variable is observed: $V = V^{\mathrm{obs}}$, $V^{\mathrm{lat}} = \emptyset$. The case with hidden variables is sketched in §6 and flagged as open.

We retain v1 conventions C1–C4 (`FOUNDATIONS.md` §0) and the standard noise-independence assumption: $u_{m_1} \perp u_{m_2}$ for $m_1 \neq m_2$.

### Two intervention regimes

In a HADMG, two natural intervention regimes arise:

- **Variable interventions** $\mathrm{do}(X = x)$ for $X \subseteq V$. These are Pearl-style; identifiability behavior matches the bipartite-blowup ADMG (Theorem T5).
- **Mechanism interventions** $\mathrm{do}(\neg m)$ and $\mathrm{do}(m \to m')$ for $m \in E$. These are the new operations; identifiability is governed by Theorem T4.

---

## 1. Theorem T4: mechanism-deletion identifiability

**Theorem T4.** Let $\mathcal{M}$ be a HADMG with $V^{\mathrm{lat}} = \emptyset$ (all variables observed). Let $m^\star \in E$ be any mechanism — observed or latent. Then $\mathrm{do}(\neg m^\star)$ is identifiable from $P(V)$ via:

$$
P^{\mathcal{M}^{\neg m^\star}}(V) \;=\; \frac{P(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v).
$$

The mechanism factor $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is a conditional distribution in $P(V)$ — directly observable, regardless of whether $m^\star$ is latent or observed.

### Proof

Apply the mechanism-level chain rule (Lemma 1.1 of `THEOREM_T2_T3.md`) to the joint distribution induced by $\mathcal{M}$:

$$
P(V) = \prod_{v \in V^{\mathrm{exo}}} P(v) \cdot \prod_{m \in E} P(\mathrm{out}(m) \mid \mathrm{in}(m)).
$$

Lemma 1.1's proof did not require observability of $f_m$ — only C1–C4 and noise independence. It applies verbatim to HADMGs. Each factor $P(\mathrm{out}(m) \mid \mathrm{in}(m))$ is a well-defined conditional in $P(V)$ since both $\mathrm{out}(m)$ and $\mathrm{in}(m)$ are observed (by $V^{\mathrm{lat}} = \emptyset$).

By the proof of T2 (`THEOREM_T2_T3.md` §2), $\mathrm{do}(\neg m^\star)$ removes the $m^\star$ factor from the product and inserts $P_0(v)$ factors for each orphaned output. The arithmetic is identical:

$$
P^{\neg m^\star}(V) = \frac{P(V)}{P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v).
$$

**Why this is identifiable.** The numerator $P(V)$ is the observational distribution. The denominator $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is read off as a conditional in the observational distribution. The $P_0$ factors are part of the model specification. No part of the formula refers to $f_m$, $u_m$, or any quantity depending on whether $m^\star$ is observed or latent. □

### What just happened (and why it is surprising)

Pearl's analogous question — when is $P(V \mid \mathrm{do}(X = x))$ identifiable in a Pearl ADMG with latent confounders — has a non-trivial answer (Shpitser-Pearl 2006: identifiable iff no hedge for $(X, V \setminus X)$). The hedge criterion is NP-hard to verify in the worst case and obstructs identifiability for many natural queries.

T4 says: in the hypergraph framework, the analogous question for mechanism deletion has a **trivial** positive answer. **Mechanism deletion is always identifiable**, latent confounding notwithstanding, as long as the variables involved are observed.

The technical reason: Pearl variable interventions cut a *single structural equation* and require accounting for back-door paths through latents. Hypergraph mechanism interventions cut a *whole mechanism factor* in the chain rule. The factor encapsulates the joint conditional distribution of the mechanism's outputs given its inputs — which, under noise independence, equals the corresponding causal mechanism factor exactly, regardless of upstream confounding. There is no back-door to close because the factorization already isolates the relevant component.

---

## 2. Theorem T5: variable-intervention identifiability via bipartite blowup

**Theorem T5.** Let $\mathcal{M}$ be a HADMG with $V^{\mathrm{lat}} = \emptyset$. Let $X, Y \subseteq V$ be disjoint. Then $P(Y \mid \mathrm{do}(X = x))$ is identifiable from $P(V)$ in $\mathcal{M}$ if and only if it is identifiable in the Pearl ADMG $B^\dagger(\mathcal{M})$ obtained from the bipartite blowup by:

1. Treating each observed mechanism node $m$ as a deterministic node with parents $\mathrm{in}(m) \cup \{u_m\}$ and children $\mathrm{out}(m)$.
2. Treating each latent mechanism node $m^{\mathrm{lat}}$ as introducing a single latent confounder $u_{m^{\mathrm{lat}}}$ with bidirected edges (Pearl-style) connecting its outputs.

The standard Shpitser-Pearl hedge criterion applied to $B^\dagger(\mathcal{M})$ yields the answer.

### Proof sketch

Soundness: any identifying expression for $P(Y \mid \mathrm{do}(X = x))$ in $B^\dagger(\mathcal{M})$ is, after the standard sampling-equivalence argument (`THEOREM_T1.md` Lemma 2.2), an identifying expression in $\mathcal{M}$ — the marginal of $\tilde{B}(\mathcal{M})$'s distribution on $V$ matches $P^{\mathcal{M}}(V)$.

Completeness: a hedge in $B^\dagger(\mathcal{M})$ is a hedge in the underlying Pearl ADMG governing $V$'s marginal distribution, and Shpitser-Pearl 2006 establishes that hedges are obstructions to identifiability in Pearl ADMGs.

The reduction is faithful because no part of the hedge construction interacts with the bipartite alternation of variable and mechanism nodes — paths in $B^\dagger$ between variable nodes either pass through mechanism nodes (treated as fixed deterministic relays) or through bidirected hyperedges from latent mechanisms (treated as Pearl bidirected edges among outputs).

The full proof is technical but follows standard Pearl ADMG machinery; we elide details here as they replicate Shpitser-Pearl 2006 §3 verbatim under the bipartite reduction. □

### Practical consequence

Identifying variable interventions in a HADMG is computationally equivalent to the Pearl ADMG identification problem. The standard ID algorithm (Shpitser-Pearl 2006) runs on $B^\dagger(\mathcal{M})$ and either returns an identifying expression or a hedge witness.

T5 is a **reduction**, not a new theorem. It says: the hypergraph framework does not require a new identifiability theory for variable interventions — Pearl's existing theory applies via the bipartite blowup.

---

## 3. The asymmetry between mechanism and variable interventions

Combining T4 and T5:

| Intervention type | Identifying expression under v1 + $V^{\mathrm{lat}} = \emptyset$ |
|---|---|
| $\mathrm{do}(\neg m)$ — mechanism deletion | **Closed form (T4):** $P(V) / P(\mathrm{out}(m) \mid \mathrm{in}(m)) \cdot \prod P_0$. Always identifiable; computable in $O(\lvert V \rvert + \lvert E \rvert)$. |
| $\mathrm{do}(m \to m')$ — mechanism replacement | **Closed form (T3 + T4):** same denominator, replacement factor in numerator. |
| $\mathrm{do}(v = x)$ — variable intervention | **Reduction to Pearl multi-variable ID on $B^\dagger(\mathcal{M})$ (T5).** No new theory required. |

The asymmetry is best stated precisely. It has two distinct components:

**(a) Closed-form vs case-analytic.** Mechanism interventions admit a single explicit identifying formula readable directly from the chain rule. Variable interventions require running Pearl's ID algorithm on the bipartite-blowup ADMG; the algorithm always terminates, but its output is case-analytic rather than uniform.

**(b) Robustness to hidden mechanisms.** T4's formula is invariant to whether $m^\star$ is observed or latent — only the typed incidence is needed. Variable-intervention identifiability via T5 depends on the specific bidirected-edge structure induced by latent mechanisms in $B^\dagger(\mathcal{M})$.

A natural further question is whether T5 ever yields *unidentifiable* on a HADMG satisfying v1 conventions (C1–C4 + $V^{\mathrm{lat}} = \emptyset$). Under HSCM intervention semantics ($\mathrm{do}(v=x)$ deletes the producing mechanism and orphans siblings to $P_0$), an intervention on any output of a multi-output mechanism corresponds in $B^\dagger(\mathcal{M})$ to a multi-variable intervention spanning the entire output set. This severs all bidirected edges that the latent mechanism would otherwise contribute, eliminating the standard Shpitser-Pearl hedge constructions. We have not been able to exhibit a v1 HADMG on which a variable intervention is genuinely unidentifiable; we conjecture none exist, but state this as observation rather than theorem and do not commit to the claim.

Under hidden *variables* ($V^{\mathrm{lat}} \neq \emptyset$, treated in `THEOREM_H1_PLUS.md`), the asymmetry sharpens: T6 settles the observed-boundary case in closed form, while T7's boundary-violating case reduces to a Pearl ID problem that *can* genuinely fail (hyper-hedge obstructions).

**Implication for the framework's value proposition.** The contribution is not new identifiability where Pearl's machinery fails — it is (i) a closed-form identifier requiring no algorithmic search, (ii) an intervention vocabulary aligned with experimental practice (one drug ablation = one $\mathrm{do}(\neg m)$, not a multi-variable Pearl translation), and (iii) under hidden variables, a clean sufficient condition (T6) that bypasses the hyper-hedge analysis entirely.

---

## 4. Worked instance: latent confounding of the reaction-network inputs

Extend the reaction network with a latent mechanism $m^{\mathrm{lat}}$ that produces both $B$ and $E$:

- $m^{\mathrm{lat}}: \emptyset \to B, E$ (latent, joint output)
- $m_1: A, B \to C, D$ (observed)
- $m_2: C, E \to F$ (observed)
- $A$ exogenous; $B, E$ confounded via $m^{\mathrm{lat}}$.

**Observational distribution.** By Lemma 1.1 (HADMG version):

$$
P(V) = P(A) \cdot P(B, E) \cdot P(C, D \mid A, B) \cdot P(F \mid C, E)
$$

where $P(B, E)$ is the joint marginal induced by $m^{\mathrm{lat}}$ (we observe it but cannot factor it as $P(B) \cdot P(E)$ since $m^{\mathrm{lat}}$ couples them).

**Apply T4 to $\mathrm{do}(\neg m_1)$.**

The mechanism factor $P(C, D \mid A, B)$ is observable as a conditional in $P(V)$. T4 gives:

$$
P^{\neg m_1}(V) = \frac{P(V)}{P(C, D \mid A, B)} \cdot \delta_0(C) \delta_0(D) = P(A) \cdot P(B, E) \cdot \delta_0(C) \delta_0(D) \cdot P(F \mid C, E).
$$

Substituting $C = 0$ in the surviving factor:

$$
P^{\neg m_1}(V) = P(A) \cdot P(B, E) \cdot \delta_0(C) \delta_0(D) \cdot \mathcal{N}(F; 0, \sigma_2^2).
$$

This is exactly what direct simulation of $\mathcal{M}^{\neg m_1}$ produces (verified numerically by `test_hadmg.py::test_T4_under_latent_confounding`). The latent confounding between $B$ and $E$ does not obstruct identification — it is preserved in the surviving $P(B, E)$ factor and propagates correctly to the post-intervention distribution.

**Apply T4 to $\mathrm{do}(\neg m^{\mathrm{lat}})$ — intervening on the latent.**

Even though $f_{m^{\mathrm{lat}}}$ is unknown, T4 still applies. The mechanism factor $P(B, E)$ is observable as the joint marginal. T4 gives:

$$
P^{\neg m^{\mathrm{lat}}}(V) = \frac{P(V)}{P(B, E)} \cdot \delta_0(B) \delta_0(E) = P(A) \cdot \delta_0(B) \delta_0(E) \cdot P(C, D \mid A, B = 0) \cdot P(F \mid C, E = 0).
$$

This is the post-intervention distribution under "deletion of the latent mechanism" — a query that has no direct Pearl analogue (Pearl latents are unaddressable). The hypergraph framework gives it an identifiable closed form.

---

## 5. Computational complexity

T4's identifying expression is computable in $O(|V| + |E|)$ time on a HADMG: find the target mechanism, read off its incidence, retrieve $P_0$. There is no analogue of Pearl's ID-algorithm recursion or hedge search.

T5's identifying expression is computable in time matching Shpitser-Pearl's ID algorithm on $B^\dagger(\mathcal{M})$ — polynomial in $|V| + |E|$ for fixed-size hedge searches, NP-hard in adversarial cases.

The computational asymmetry mirrors the theoretical asymmetry of §3: mechanism-level identifiability is *trivially easy*; variable-level is *as hard as Pearl*.

---

## 6. Open: identifiability under hidden variables

The v1 result (T4) requires $V^{\mathrm{lat}} = \emptyset$ — every variable observed. Allowing hidden variables (some $v \in V \setminus V^{\mathrm{obs}}$) introduces genuine obstructions:

- If $\mathrm{in}(m^\star)$ contains a hidden variable, the conditional $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is not directly observable; it must be reconstructed from $P(V^{\mathrm{obs}})$ via marginalization, which may or may not be uniquely possible.
- If $\mathrm{out}(m^\star)$ contains a hidden variable, the post-intervention distribution over observed variables involves an expectation over the unobserved output, which may be inestimable.

We conjecture:

**Conjecture H1+ (hyper-hedge characterization).** $\mathrm{do}(\neg m^\star)$ is identifiable from $P(V^{\mathrm{obs}})$ in a general HADMG (with possibly hidden variables) if and only if there is no **hyper-hedge** for $m^\star$ — a structural obstruction defined by analogy to Pearl's hedge but accounting for joint mechanism factors. Stating this precisely and proving it is open.

A concrete approach: represent the HADMG as a Pearl ADMG with mechanism nodes, augmented with deterministic-relations annotations encoding joint output structure. Apply Shpitser-Pearl's ID algorithm to this augmented ADMG, with hedge search modified to respect the deterministic annotations. Whether the resulting criterion is sound and complete in the hypergraph sense remains to be verified.

---

## 7. Relationship to T2/T3

T2 and T3 (`THEOREM_T2_T3.md`) assumed full causal sufficiency — no latent mechanisms. T4 strictly subsumes T2:

- T2 + causal sufficiency $\Rightarrow$ T4 with $E^{\mathrm{lat}} = \emptyset$.
- T4 with $E^{\mathrm{lat}} = \emptyset$ $\Leftrightarrow$ T2.
- T4 strictly extends T2 to allow $E^{\mathrm{lat}} \neq \emptyset$.

T3 (mechanism replacement) likewise generalizes: under v1 with all-variables-observed, $\mathrm{do}(m^\star \to m')$ is identifiable for any (observed or latent) target $m^\star$, with the formula

$$
P^{m^\star \to m'}(V) = \frac{P(V)}{P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))} \cdot P_{f_{m'}}(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)).
$$

provided the replacement's induced distribution $P_{f_{m'}}$ is specified.

We do not state this as a separate theorem but include it as **Corollary T4.1**.

---

## 8. Mechanism-correlated noise: scope of T4

T4's derivation depends on the cross-mechanism noise-independence assumption $u_{m_1} \perp u_{m_2}$ (in the spirit of C2). The step "the conditional $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ equals the causal mechanism factor" requires $u_{m^\star} \perp \mathrm{in}(m^\star)$, which holds under noise independence.

If this assumption is relaxed — mechanisms share correlated noise via shared environmental input or common stochastic source — the mechanism factor in $P(V)$ ceases to equal the pushforward of $u_{m^\star}$ through $f_{m^\star}(\mathrm{in}(m^\star), \cdot)$, and T4's formula no longer applies. An earlier draft conjectured (**H2**) that this regime exhibits conditional independencies inexpressible by any finite-latent Pearl SCM — i.e., representational strict-dominance of the hypergraph framework. We have **retracted H2** (`THEOREM_T2_T3.md` §5.2): Pearl-with-latents is universal at both the distributional and interventional level, so any HSCM with correlated mechanism noise has a Pearl realization with appropriately introduced latent common causes.

The substantive consequence: under correlated mechanism noise, identifiability of $\mathrm{do}(\neg m^\star)$ falls back to standard hidden-confounder analysis on a Pearl ADMG obtained from the bipartite blowup (T7's machinery, applied with the correlation latents made explicit). There is no special hypergraph hedge structure beyond the Pearl one. The framework's value under correlated noise is therefore modeling-ergonomic — keeping correlation at the mechanism level rather than reifying it as Pearl latents — not theoretical strict-dominance.

The roadmap that remains:

- **T4** (proved) — closes the easy case (v1 + noise independence): mechanism deletion identifiable in HADMGs with $V^{\mathrm{lat}} = \emptyset$.
- **T6** (proved, `THEOREM_H1_PLUS.md`) — extends T4 to hidden-variable HADMGs under the observed-boundary condition.
- **T7** (proof-sketched) — reduces the boundary-violating case to Pearl ADMG identification.
- **H1+ completeness** (open) — the converse of T7's reduction, lifting Shpitser-Pearl 2006 through the bipartite blowup.

---

## References

- Shpitser, I. & Pearl, J. (2006). "Identification of conditional interventional distributions." *UAI 22*.
- Tian, J. & Pearl, J. (2002). "A general identification condition for causal effects." *AAAI 18*.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press, Ch. 3.
- Bareinboim, E. & Pearl, J. (2016). "Causal inference and the data-fusion problem." *PNAS* 113.
