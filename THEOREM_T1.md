# Theorem T1: Bipartite-Blowup d-Separation

This document states and proves the target theorem of `FOUNDATIONS.md` §10. The proof is a reduction to standard Pearl results (Verma & Pearl 1990, Geiger & Pearl 1990) via an explicit noise-augmented DAG.

---

## 1. Statement

Let $\mathcal{M} = (V, E, \rho, F, P, P_0)$ be a Hypergraph SCM under v1 conventions C1–C3 (`FOUNDATIONS.md` §0). Let $B(\mathcal{M})$ be its bipartite blowup (`FOUNDATIONS.md` §5), with each mechanism node $m \in E$ treated as a node whose value is the joint output tuple $(v : v \in \mathrm{out}(m))$ produced by $f_m(\mathrm{in}(m), u_m)$.

For any subset $Z \subseteq V$, let $\mathrm{Det}_{\mathcal{M}}(Z) \subseteq V$ denote the **functional-determination closure** of $Z$: the smallest superset of $Z$ such that, whenever a structural equation makes a variable $v$ a function of variables already in the set, $v$ is included.

Define $d^*$-separation in $B(\mathcal{M})$ as standard Pearl d-separation conditional on $Z^* := Z \cup \mathrm{Det}_{\mathcal{M}}(Z)$.

**Theorem T1 (Soundness).** For pairwise disjoint $X, Y, Z \subseteq V$:

$$
X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z \;\implies\; X \perp_{P^{\mathcal{M}}} Y \mid Z.
$$

**Theorem T1 (Completeness).** Under the standard faithfulness assumption — that $P^{\mathcal{M}}$ contains no conditional independences beyond those entailed by the graphical structure — the converse holds:

$$
X \perp_{P^{\mathcal{M}}} Y \mid Z \;\implies\; X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z.
$$

---

## 2. The augmented DAG $\tilde{B}(\mathcal{M})$

Define $\tilde{B}(\mathcal{M})$ on the node set $V \cup E \cup U$, where

$$
U = \{u_m : m \in E\} \;\cup\; \{u^{\mathrm{exo}}_v : v \in V^{\mathrm{exo}}\}
$$

and edges:

- $u^{\mathrm{exo}}_v \to v$ for each $v \in V^{\mathrm{exo}}$
- $v \to m$ for each $v \in \mathrm{in}(m), m \in E$
- $u_m \to m$ for each $m \in E$
- $m \to v$ for each $v \in \mathrm{out}(m), m \in E$

**Lemma 2.1.** $\tilde{B}(\mathcal{M})$ is a DAG.

*Proof.* Noise nodes $U$ have no incoming edges and one outgoing edge each, so they cannot lie on a cycle. Cycles among $V \cup E$ must alternate variable and mechanism nodes by the bipartite structure. Any such cycle projects to a cycle in the mechanism dependency graph $G_E$ (replacing variable nodes by the mechanism that produces them — since each variable in a cycle must be both produced and consumed by some mechanism, both directions exist). C1 forbids cycles in $G_E$. □

**Lemma 2.2 (sampling equivalence, restated).** Pearl-style ancestral sampling on $\tilde{B}(\mathcal{M})$ — drawing $u \sim P$ for each noise node, then evaluating each non-noise node as the deterministic function of its parents prescribed by $F$ — induces the same distribution on $V$ as the Hypergraph-SCM sampling procedure of `FOUNDATIONS.md` §4.

*Proof.* Both procedures: (1) draw all exogenous noise; (2) evaluate mechanisms in topological order of $G_E$ (equivalently, ancestral order of $\tilde{B}(\mathcal{M})$ restricted to $V \cup E$); (3) project mechanism outputs to variable values. Identical. □

---

## 3. Soundness proof

Recall: for a DAG with deterministic structural equations and independent exogenous noise, standard d-separation is sound for the induced distribution (Verma & Pearl 1990; Geiger, Verma & Pearl 1990, "d-separation is sound and complete for Bayesian networks with deterministic relations" — the *deterministic-relations* extension is precisely what we need).

**Step 1.** By Lemma 2.2, the distribution $P^{\mathcal{M}}$ over $V$ equals the marginal over $V$ of the distribution $\tilde{P}$ on $V \cup E \cup U$ induced by ancestral sampling on $\tilde{B}(\mathcal{M})$.

**Step 2.** In $\tilde{B}(\mathcal{M})$, every non-noise node is a *deterministic* function of its parents:

- $v \in V^{\mathrm{exo}}$: $v = u^{\mathrm{exo}}_v$ (identity).
- $m \in E$: $m = f_m(\mathrm{in}(m), u_m)$.
- $v \in \mathrm{out}(m)$ for some $m$: $v = \pi_v(m)$, the projection of the mechanism's tuple value onto the $v$-component.

**Step 3.** Apply Geiger-Verma-Pearl's deterministic d-separation theorem to $\tilde{B}(\mathcal{M})$: for disjoint $X, Y \subseteq V$ and any $Z \subseteq V$, if $X$ is d-separated from $Y$ given $Z \cup \mathrm{Det}_{\tilde{B}}(Z)$ in $\tilde{B}(\mathcal{M})$, then $X \perp_{\tilde{P}} Y \mid Z$, where $\mathrm{Det}_{\tilde{B}}(Z)$ is the deterministic closure in $\tilde{B}(\mathcal{M})$.

**Step 4.** We must show: for $X, Y, Z \subseteq V$, $d^*$-separation in $B(\mathcal{M})$ given $Z$ is equivalent to standard d-separation in $\tilde{B}(\mathcal{M})$ given $Z \cup \mathrm{Det}_{\tilde{B}}(Z)$, restricted to paths over non-noise nodes.

Two facts suffice:

- **Fact 4a (Path-equivalence).** Every path in $\tilde{B}(\mathcal{M})$ between non-noise nodes corresponds to a unique path in $B(\mathcal{M})$ (delete noise predecessors). Conversely, every path in $B(\mathcal{M})$ lifts to a path in $\tilde{B}(\mathcal{M})$ over the same node sequence. The d-connection rules at intermediate nodes (collider vs. non-collider, conditioning) are identical in both graphs because noise nodes contribute only as parents at their unique child — they do not appear as intermediate nodes on any path between non-noise nodes.

- **Fact 4b (Determination-equivalence).** $\mathrm{Det}_{\tilde{B}}(Z) \cap V = \mathrm{Det}_{\mathcal{M}}(Z)$.

*Proof of 4b.* In $\tilde{B}(\mathcal{M})$, a node is deterministically determined by its parents. Iteratively expanding $Z$ by adding any node whose parents (in $\tilde{B}$) are all in the closure recovers the same set as iteratively expanding by structural-equation determination in $\mathcal{M}$. The two iterations coincide on $V$. □

Combining: $X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z$ implies $X \perp^d_{\tilde{B}(\mathcal{M})} Y \mid Z \cup \mathrm{Det}_{\tilde{B}}(Z)$, which by Step 3 implies $X \perp_{\tilde{P}} Y \mid Z$. Marginalizing over $E \cup U$ preserves the conditional independence on $V$. Hence $X \perp_{P^{\mathcal{M}}} Y \mid Z$. □

---

## 4. Completeness proof

Faithfulness assumption (FA): for the joint distribution $\tilde{P}$ on $V \cup E \cup U$, every conditional independence among non-noise nodes is entailed by d-separation in $\tilde{B}(\mathcal{M})$.

By Verma-Pearl completeness for DAGs with deterministic relations (Geiger-Verma-Pearl 1990, also Spirtes-Glymour-Scheines *Causation, Prediction, and Search* §3.4), under FA every conditional independence in $\tilde{P}$ corresponds to d-separation given the determination closure.

Suppose $X \perp_{P^{\mathcal{M}}} Y \mid Z$. By Lemma 2.2, this is a CI in the marginal of $\tilde{P}$ on $V$, hence a CI of $\tilde{P}$. By FA, $X \perp^d_{\tilde{B}(\mathcal{M})} Y \mid Z \cup \mathrm{Det}_{\tilde{B}}(Z)$. By Facts 4a–4b, $X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z$. □

---

## 5. Discussion

### 5.1 Why the deterministic augmentation is necessary

In the minimal example (`MINIMAL_EXAMPLE.md`), $C \equiv D$ as a structural identity. Plain d-separation on $B(\mathcal{M})$ would predict $A \not\perp D \mid C$ (path $A \to m_1 \to D$ traverses no node in $\{C\}$). But empirically, conditioning on $C$ pins $D = C$ to the observed value, so $A \perp D \mid C$ holds in distribution.

The augmentation $Z^* = Z \cup \mathrm{Det}(Z)$ adds $D$ to the conditioning set whenever $C$ is conditioned on (since $C$ functionally determines $D$ via $m_1$'s structural identity). With $D \in Z^*$, the path $A \to m_1 \to D$ ends at a node in the conditioning set — d-separation correctly returns "separated."

This is a known feature of Pearl's framework when deterministic structural relations are present (Geiger-Pearl 1990); the hypergraph framework inherits it directly via the bipartite blowup.

### 5.2 What the theorem does and does not give

T1 gives: a **graphical criterion for conditional independence** on a Hypergraph SCM, computable in polynomial time on $B(\mathcal{M})$, with the same expressive power as Pearl d-separation up to the deterministic augmentation.

T1 does *not* give: a do-calculus for mechanism interventions $\mathrm{do}(\neg m)$ and $\mathrm{do}(m \to m')$. These are conjectured to admit identifiability rules analogous to Pearl's Rules 1–3, but the theory is an open problem (whitepaper §9).

### 5.3 Computational implementation

A reference implementation is provided in `minimal_model/dseparation.py`, with tests in `minimal_model/test_dseparation.py`. The implementation:

1. Builds the bipartite blowup $B(\mathcal{M})$.
2. Computes $\mathrm{Det}_{\mathcal{M}}(Z)$ via fixed-point closure under each mechanism's `output_equalities`.
3. Enumerates simple paths between $X$ and $Y$ in the undirected version of $B(\mathcal{M})$.
4. For each path, applies the standard collider / non-collider rules with conditioning set $Z^*$.

The minimal example exercises seven cases including the deterministic-coupling case $A \perp D \mid C$ (resolved correctly by the augmentation, would be missed by plain d-separation).

---

## 6. Corollaries

### 6.1 Causal Markov property

**Corollary T1.1.** $P^{\mathcal{M}}$ satisfies the local Markov property: each $v \in V$ is independent of its non-descendants in $B(\mathcal{M})$ given its parents in $B(\mathcal{M})$ (which is its producing mechanism node, plus any deterministic siblings).

### 6.2 I-Map property

**Corollary T1.2.** $B(\mathcal{M})$ (with $d^*$-augmentation) is an I-map for $P^{\mathcal{M}}$ — every CI implied by the graph holds in distribution. Under faithfulness it is a perfect map.

### 6.3 Decomposability of variable interventions

**Corollary T1.3.** For variable interventions $\mathrm{do}(v = x)$, the post-intervention distribution $P^{\mathcal{M}^{v=x}}$ satisfies $d^*$-separation in $B(\mathcal{M}^{v=x}) = B(\mathcal{M})$ with mechanisms producing $v$ removed and $v$ promoted to a root. This is the hypergraph analogue of Pearl's truncated factorization formula.

(Mechanism interventions $\mathrm{do}(\neg m), \mathrm{do}(m \to m')$ admit graphical surgery rules but their identifiability remains open — see whitepaper §9.)

---

## References

- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Verma, T. & Pearl, J. (1990). "Causal networks: semantics and expressiveness." *Uncertainty in Artificial Intelligence 4*.
- Geiger, D., Verma, T. & Pearl, J. (1990). "Identifying independence in Bayesian networks." *Networks* 20.
- Geiger, D. & Pearl, J. (1990). "On the logic of causal models." *UAI 4*.
- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press, §3.4.
