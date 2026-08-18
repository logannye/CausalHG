# Theorem T1: Bipartite-Blowup d-Separation

This document states and proves the target theorem of `FOUNDATIONS.md` §10. The proof is a reduction to standard Pearl results (Verma & Pearl 1990, Geiger & Pearl 1990) via an explicit noise-augmented DAG.

---

## 1. Statement

Let $\mathcal{M} = (V, E, \rho, F, P, P_0)$ be a Hypergraph SCM under v1 conventions C1–C3 (`FOUNDATIONS.md` §0). Let $B(\mathcal{M})$ be its bipartite blowup (`FOUNDATIONS.md` §5), with each mechanism node $m \in E$ treated as a node whose value is the joint output tuple $(v : v \in \mathrm{out}(m))$ produced by $f_m(\mathrm{in}(m), u_m)$.

### 1.1 Two determination sets, kept apart

The criterion depends on a notion of "already pinned down by $Z$", and it is essential to distinguish the *true* such set from the one the criterion can actually compute.

**Definition (true determination set).** For $Z \subseteq V$, let

$$
D_{\mathcal{M}}(Z) \;=\; \{\, v \in V : v = g(Z) \text{ } P^{\mathcal{M}}\text{-a.s. for some measurable } g \,\}.
$$

This is a property of the distribution, not of the incidence, and is not computable from $\rho$ alone.

**Definition (declared determination rules).** A rule set $R$ is a finite family of closure operators on subsets of $V$. The **declared closure** $\mathrm{Det}_R(Z)$ is the least fixed point containing $Z$ under $R$. In v1, $R$ consists of one rule per declared `output_equalities` group $G$: if $G \cap S \neq \emptyset$ then $G \subseteq S$.

**Definition (validity).** $R$ is *valid* for $\mathcal{M}$ if $\mathrm{Det}_R(Z) \subseteq D_{\mathcal{M}}(Z)$ for every $Z \subseteq V$ — every variable the rules declare determined really is. For output-equality groups this holds exactly when the declared members are $P^{\mathcal{M}}$-a.s. equal, which is what `output_equalities` asserts.

Validity is a hypothesis on the *model description*, not something the compiler verifies: it has no access to $F$. Declaring an equality that does not hold is garbage-in, and §5.4 exhibits the resulting unsoundness.

Define $d^*$-separation in $B(\mathcal{M})$ given $Z$ as standard Pearl d-separation of $X \setminus Z^*$ from $Y \setminus Z^*$ conditional on $Z^* := \mathrm{Det}_R(Z)$, where a query with $X \subseteq Z^*$ or $Y \subseteq Z^*$ counts as separated.

### 1.2 The theorem

**Theorem T1 (Soundness).** Let $R$ be valid for $\mathcal{M}$. For pairwise disjoint $X, Y, Z \subseteq V$:

$$
X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z \;\implies\; X \perp_{P^{\mathcal{M}}} Y \mid Z.
$$

Note what this does **not** require: no relationship between $\mathrm{Det}_R$ and $D_{\mathcal{M}}$ beyond containment. Soundness survives however incomplete the declared rules are.

**Theorem T1 (Completeness).** Assume in addition

- **(FA)** faithfulness: $P^{\mathcal{M}}$ contains no conditional independences beyond those entailed by the graphical structure of $\tilde{B}(\mathcal{M})$; and
- **(DC)** declaration completeness: $\mathrm{Det}_R(Z) = D_{\mathcal{M}}(Z)$ for every $Z \subseteq V$.

Then the converse holds:

$$
X \perp_{P^{\mathcal{M}}} Y \mid Z \;\implies\; X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z.
$$

(DC is a genuine restriction. Functional determination can arise from sources v1 does not let you declare — an injective $f_m$ makes $\mathrm{in}(m)$ recoverable from $\mathrm{out}(m)$, and structural zeros in a kernel create determination with no equality group behind it. Where DC fails, completeness fails **one-sidedly**: independences are missed, never falsely claimed. §5.4 quantifies this.)

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

The argument uses only *ordinary* d-separation soundness on $\tilde{B}(\mathcal{M})$ (Verma & Pearl 1990). The deterministic-relations extension of Geiger-Verma-Pearl is needed for **completeness** (§4), not here — a point worth making explicitly, because it is what lets soundness survive an incomplete rule set.

Write $Z^* := \mathrm{Det}_R(Z)$ throughout, and note $Z \subseteq Z^*$.

**Step 1.** By Lemma 2.2, $P^{\mathcal{M}}$ is the $V$-marginal of the distribution $\tilde{P}$ on $V \cup E \cup U$ induced by ancestral sampling on $\tilde{B}(\mathcal{M})$. By Lemma 2.1 $\tilde{B}(\mathcal{M})$ is a DAG, and ancestral sampling makes $\tilde{P}$ Markov with respect to it. Hence ordinary d-separation in $\tilde{B}(\mathcal{M})$ implies conditional independence in $\tilde{P}$, **for any conditioning set**.

**Step 2 (Fact 4a, path-equivalence).** Every path in $\tilde{B}(\mathcal{M})$ between non-noise nodes corresponds to a unique path in $B(\mathcal{M})$ (delete noise predecessors), and conversely. Noise nodes have no parents and exactly one child, so a path reaching $u$ cannot leave it: noise nodes are never *intermediate* nodes on a path between non-noise nodes, only endpoints. Since $X, Y, Z^* \subseteq V$, no such path is at issue. Collider status at an intermediate mechanism node $m$ is also unchanged, because the only extra parent $u_m$ never lies on the path, and the descendant sets of non-noise nodes agree in the two graphs.

Consequently $X \setminus Z^* \perp^{d}_{B(\mathcal{M})} Y \setminus Z^* \mid Z^*$ implies the same d-separation in $\tilde{B}(\mathcal{M})$.

**Step 3.** By Step 2 the separation holds in $\tilde{B}(\mathcal{M})$, so by Step 1

$$
(X \setminus Z^*) \;\perp_{\tilde{P}}\; (Y \setminus Z^*) \;\bigm|\; Z^*.
$$

All three sets lie in $V$, so this is a statement about the $V$-marginal, and by Lemma 2.2 that marginal is $P^{\mathcal{M}}$:

$$
(X \setminus Z^*) \;\perp_{P^{\mathcal{M}}}\; (Y \setminus Z^*) \;\bigm|\; Z^*.
$$

**Step 4 (discharging the augmentation).** Here validity of $R$ enters, and only here. Validity says $Z^* = \mathrm{Det}_R(Z) \subseteq D_{\mathcal{M}}(Z)$, so every $c \in Z^* \setminus Z$ satisfies $c = g_c(Z)$ a.s. for some measurable $g_c$, whence

$$
\sigma(Z^*) \;=\; \sigma\!\left(Z \cup \{g_c(Z)\}_c\right) \;=\; \sigma(Z) \quad\text{up to } P^{\mathcal{M}}\text{-null sets}.
$$

Conditioning on $Z^*$ is therefore conditioning on $Z$, and Step 3 becomes

$$
(X \setminus Z^*) \;\perp_{P^{\mathcal{M}}}\; (Y \setminus Z^*) \;\bigm|\; Z.
$$

**Step 5 (restoring the determined coordinates).** For $P_Z$-a.e. $z$, each $x \in X \cap Z^*$ equals $g_x(z)$ almost surely and is therefore degenerate under the regular conditional distribution $P(\cdot \mid Z = z)$; likewise for $Y \cap Z^*$. A degenerate coordinate is independent of every other random element, so under $P(\cdot \mid Z = z)$ the law of $X$ is a point mass on the $X \cap Z^*$ coordinates times the law of $X \setminus Z^*$, and similarly for $Y$. Combining with Step 4,

$$
X \perp_{P^{\mathcal{M}}} Y \mid Z. \qquad \square
$$

The degenerate case is included: if $X \subseteq Z^*$ then $X$ is a.s. constant given $Z$ and the conclusion is immediate, which is why the criterion in §1.1 counts that query as separated.

**Remark 3.1 (what replaced Fact 4b).** Earlier drafts asserted a *Determination-equivalence* fact, $\mathrm{Det}_{\tilde{B}}(Z) \cap V = \mathrm{Det}_{\mathcal{M}}(Z)$, proved by iteratively adding nodes all of whose parents are known. That equality is false as stated, and the counterexample is this framework's own worked example. With $C \equiv D$ produced jointly by $m_1$ and $Z = \{C\}$: the parent iteration cannot add $D$, since $D$'s only parent $m_1$ is not in the closure and $m_1$'s parents $\{A, B, u_{m_1}\}$ are not either — it halts at $\{C\}$. But $C$ does determine $D$. The iteration propagates determination only *downward*, from parents to child, whereas $C \equiv D$ is a *sibling* relation requiring it to travel up into $m_1$ and back down, which is exactly the reasoning §5.1 describes informally.

The repair is not to fix that iteration but to observe that soundness never needed it. Steps 1–5 use only ordinary d-separation plus validity of $R$; the size of $\mathrm{Det}_R$ relative to $D_{\mathcal{M}}$ affects how *much* the criterion can prove, not whether what it proves is true.

---

## 4. Completeness proof

This is the direction that needs the deterministic-relations machinery, and the direction where the strength of $R$ becomes load-bearing.

Recall the two hypotheses from §1.2: **(FA)** faithfulness for $\tilde{P}$, and **(DC)** $\mathrm{Det}_R(Z) = D_{\mathcal{M}}(Z)$ for every $Z$.

By Geiger-Verma-Pearl completeness for DAGs with deterministic relations (also Spirtes-Glymour-Scheines §3.4), under FA every conditional independence in $\tilde{P}$ corresponds to d-separation given the conditioning set augmented by its *true* determination closure.

Suppose $X \perp_{P^{\mathcal{M}}} Y \mid Z$. By Lemma 2.2 this is a CI of $\tilde{P}$. By FA it is witnessed by d-separation in $\tilde{B}(\mathcal{M})$ given $Z$ augmented by $D_{\mathcal{M}}(Z)$. By DC that augmented set is exactly $\mathrm{Det}_R(Z) = Z^*$, and by Fact 4a the separation transfers to $B(\mathcal{M})$. Hence $X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z$. $\square$

**Remark 4.1 (what DC costs, and which way).** DC is where the old Fact 4b was really doing its work, and it is an assumption rather than a lemma. If $\mathrm{Det}_R(Z) \subsetneq D_{\mathcal{M}}(Z)$ — some determination holds in the model but is not declared — the criterion conditions on a smaller set, blocks fewer paths, and reports fewer separations. It cannot report *more*: soundness (§3) never referred to $D_{\mathcal{M}}$ at all.

So the two halves of T1 degrade very differently, and only one of them can hurt a caller who trusts a verdict. That asymmetry is the design property worth relying on: **the oracle's failure mode under an incomplete rule set is refusal, not error.**

---

## 5. Discussion

### 5.1 Why the deterministic augmentation is necessary

In the minimal example (`MINIMAL_EXAMPLE.md`), $C \equiv D$ as a structural identity. Plain d-separation on $B(\mathcal{M})$ would predict $A \not\perp D \mid C$ (path $A \to m_1 \to D$ traverses no node in $\{C\}$). But empirically, conditioning on $C$ pins $D = C$ to the observed value, so $A \perp D \mid C$ holds in distribution.

The augmentation $Z^* = \mathrm{Det}_R(Z)$ adds $D$ to the conditioning set whenever $C$ is conditioned on, via the declared equality group $\{C, D\}$ on $m_1$. With $D \in Z^*$, the query becomes $A$ versus $Y \setminus Z^* = \emptyset$, and §1.1 counts it as separated — which is correct, since $D$ is a.s. constant given $C$.

Note that this is a *declared* rule doing the work, not an inference from incidence. The framework cannot derive $C \equiv D$ from $\rho(m_1)$ alone: incidence says $m_1$ produces $C$ and $D$ jointly, not that it produces them equal. That is exactly why §1.1 separates $\mathrm{Det}_R$ from $D_{\mathcal{M}}$, and why validity of $R$ is a hypothesis on the model description.

This is a known feature of Pearl's framework when deterministic structural relations are present (Geiger-Pearl 1990); the hypergraph framework inherits it directly via the bipartite blowup.

### 5.2 What the theorem does and does not give

T1 gives: a **graphical criterion for conditional independence** on a Hypergraph SCM, computable in polynomial time on $B(\mathcal{M})$, with the same expressive power as Pearl d-separation up to the deterministic augmentation.

T1 does *not* give: a do-calculus for mechanism interventions $\mathrm{do}(\neg m)$ and $\mathrm{do}(m \to m')$. These are conjectured to admit identifiability rules analogous to Pearl's Rules 1–3, but the theory is an open problem (whitepaper §9).

### 5.3 Computational implementation

There are two implementations, and they track the two halves of §1.1 in the same way.

`src/causal_hypergraphs/separation` is the one to use. It:

1. Builds the bipartite blowup $B(\mathcal{M})$.
2. Computes $Z^* = \mathrm{Det}_R(Z)$ as the fixed-point closure under the declared `output_equalities` rules — $\mathrm{Det}_R$, exactly, with no attempt at $D_{\mathcal{M}}$.
3. Removes the determined coordinates, returning "separated" when $X \setminus Z^*$ or $Y \setminus Z^*$ is empty (Step 5).
4. Decides the remainder by Bayes-Ball reachability given $Z^*$, visiting each (node, direction) state once, in $O(|V| + |E|)$.

`minimal_model/dseparation.py` keeps a readable path-enumeration form as the paper's appendix. It decides the same question by enumerating simple paths and applying the collider rules directly; its enumeration cap raises rather than returning a verdict, since a truncated search would report "no open path found" as separation.

The minimal example exercises seven cases including the deterministic-coupling case $A \perp D \mid C$ (resolved by the declared equality, and missed by plain d-separation).

### 5.4 Empirical status

The soundness proof of §3 is unconditional given validity of $R$; the completeness proof of §4 rests on FA and DC, neither of which is checkable. `tests/conformance/` therefore measures both directions against exact conditional independence computed from generated models' own joint laws.

| Claim | Evidence |
|---|---|
| Soundness (§3) | 5,400 $(X, Y, Z)$ triples across 120 models with positive, sparse, and singular kernels: **zero** false separations. |
| Completeness under FA + DC (§4) | On models with strictly positive kernels — no deterministic structure, so DC holds vacuously and unfaithful parameterizations are measure zero — **zero** missed independences over 5,400 triples (and over 18,000 while sizing the test). |
| Remark 4.1, DC failing one-sidedly | Withholding the equality declarations on the same models raises missed independences from 66 to 411 while leaving false separations at **zero**. Declaring them recovers 345 independences, so $R$ is not inert. |
| Validity is load-bearing (§1.1) | Declaring equalities the kernels do not satisfy produces **142** false separations across 22 of 120 models. Step 4 is therefore doing real work; the hypothesis is not decorative. |

The 66 residual misses in the mixed sweep all come from kernels with undeclared structural zeros — determination with no equality group behind it, which is precisely the DC failure Remark 4.1 anticipates.

---

## 6. Corollaries

### 6.1 Causal Markov property

**Corollary T1.1.** $P^{\mathcal{M}}$ satisfies the local Markov property: each $v \in V$ is independent of its non-descendants in $B(\mathcal{M})$ given its parents in $B(\mathcal{M})$ (which is its producing mechanism node, plus any deterministic siblings).

### 6.2 I-Map property

**Corollary T1.2.** For any valid rule set $R$, $B(\mathcal{M})$ with the $d^*$-augmentation is an I-map for $P^{\mathcal{M}}$ — every CI implied by the graph holds in distribution. Under FA **and** DC it is a perfect map. Validity alone gives the I-map property; the perfect-map property is what an incomplete $R$ costs.

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
