# Theorems T4 and T5: Identifiability Under Hidden Mechanisms

This document is the principal H1 contribution. It addresses identifiability of post-intervention distributions in **Hypergraph ADMGs (HADMGs)** — the analogue of Pearl's acyclic directed mixed graphs in the hypergraph framework.

The headline finding is *not* that the hypergraph framework needs its own version of Shpitser-Pearl 2006's hedge criterion (as originally conjectured in `THEOREM_T2_T3.md` §5.1) — it is that **mechanism-deletion identifiability is strictly easier than variable-intervention identifiability**, because mechanism-level interventions act at the level where the chain rule already factorizes. The hedge machinery is needed only for variable interventions, and there it lifts directly via the bipartite blowup.

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

Combining T4 and T5 gives a clean picture:

| Intervention type | Identifiability under hidden mechanisms |
|---|---|
| $\mathrm{do}(v = x)$ — variable intervention | Pearl hedge criterion via $B^\dagger(\mathcal{M})$ — non-trivial, may fail. |
| $\mathrm{do}(\neg m)$ — mechanism deletion | **Always identifiable** (T4) under v1 + all-variables-observed. |
| $\mathrm{do}(m \to m')$ — mechanism replacement | **Always identifiable** under same conditions (T3 + T4). |

This asymmetry is the substantive theoretical content of H1. It is not an accident of v1 conventions — it is structural. Mechanism-level interventions interact with the chain-rule factorization at the level of *factors*, where confounding is already integrated out. Variable-level interventions interact at the level of *individual structural equations*, where confounding obstructs.

**Implication for the framework's value proposition.** The hypergraph framework is not just notationally cleaner — its native intervention vocabulary (mechanism deletion, mechanism replacement) admits a strictly easier identifiability theory than Pearl's variable interventions. Modeling a domain with first-class mechanisms gains real epistemic leverage: more queries are answerable from observational data alone.

This is the strongest argument yet for why "first-class addressability of mechanisms" matters formally, not just ergonomically.

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

## 8. Implications for H2

H2 (`THEOREM_T2_T3.md` §5.2) conjectures that mechanism-correlated noise — distinct mechanisms with correlated noise — generates conditional independencies inexpressible by any finite-latent Pearl SCM. T4's derivation depended explicitly on the noise-independence assumption: the step "the conditional $P(\mathrm{out}(m) \mid \mathrm{in}(m))$ equals the causal mechanism factor" requires $u_m \perp \mathrm{in}(m)$.

Under mechanism-correlated noise, this fails. T4's clean formula ceases to apply, and a richer identifiability theory becomes necessary. **H2 is therefore the genuine frontier**: the case where mechanism interventions go from trivially-easy (T4) to as-hard-or-harder than Pearl variable interventions.

The roadmap is now explicit:

- **T4** — closes the easy case (v1 + noise independence): mechanism interventions identifiable, asymmetrically easier than variable interventions.
- **H1+** (open) — extends T4 to hidden-variable HADMGs via hyper-hedge characterization.
- **H2** (open) — establishes representational strict-dominance of the hypergraph framework via mechanism-correlated-noise expressivity.

---

## References

- Shpitser, I. & Pearl, J. (2006). "Identification of conditional interventional distributions." *UAI 22*.
- Tian, J. & Pearl, J. (2002). "A general identification condition for causal effects." *AAAI 18*.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press, Ch. 3.
- Bareinboim, E. & Pearl, J. (2016). "Causal inference and the data-fusion problem." *PNAS* 113.
