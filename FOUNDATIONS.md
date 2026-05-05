# Foundations of Causal Hypergraphs (v0.1)

This document fixes the formal substrate. Three conventions are committed in v1; alternatives are catalogued in §11.

---

## 0. Conventions committed in v1

- **C1. Mechanism-level acyclicity.** The hyperedge dependency graph (mechanism → mechanism if any output of the first is an input of the second) is a DAG.
- **C2. Deterministic structural functions with explicit noise.** Each mechanism is a deterministic map from (inputs, noise) to outputs; stochasticity is carried by independent per-mechanism noise terms.
- **C3. Bipartite role typing.** Each hyperedge has a partition of its incident nodes into inputs and outputs only. Richer roles (substrate / enzyme / product / cofactor) are out of v1 scope.
- **C4. Single producer.** For all distinct $m_1, m_2 \in E$: $\mathrm{out}(m_1) \cap \mathrm{out}(m_2) = \emptyset$. Each variable has at most one producing mechanism. This is the hypergraph analogue of Pearl's "one structural equation per variable"; it is required for the mechanism-level chain rule (Lemma 1.1 of `THEOREM_T2_T3.md`).

---

## 1. Notation

- $V = \{v_1, \ldots, v_n\}$: finite set of *variables* (nodes).
- $E = \{m_1, \ldots, m_k\}$: finite set of *mechanisms* (hyperedges).
- For each $m \in E$: $\mathrm{in}(m) \subseteq V$ and $\mathrm{out}(m) \subseteq V$ with $\mathrm{in}(m) \cap \mathrm{out}(m) = \emptyset$.
- $V^{\mathrm{exo}} = \{v \in V : \nexists m \in E,\, v \in \mathrm{out}(m)\}$: variables produced by no mechanism.
- $U = \{u_m\}_{m \in E}$: independent exogenous noise terms, one per mechanism.
- $U^{\mathrm{exo}} = \{u_v\}_{v \in V^{\mathrm{exo}}}$: exogenous noise for $V^{\mathrm{exo}}$.

The **typed incidence matrix** $M \in \{-1, 0, +1\}^{|V| \times |E|}$:

$$
M[v, m] = \begin{cases} +1 & v \in \mathrm{out}(m) \\ -1 & v \in \mathrm{in}(m) \\ 0 & \text{otherwise.} \end{cases}
$$

The standard hypergraph incidence matrix $|M|$ is recovered by $|M[v,m]| \in \{0,1\}$. The *typing* (sign) is what distinguishes a causal hypergraph from a relational hypergraph.

---

## 2. Definition (Hypergraph SCM)

A **Hypergraph SCM** is a tuple

$$
\mathcal{M} = (V,\, E,\, \rho,\, F,\, P,\, P_0)
$$

where:

- $V, E$ as above.
- $\rho: E \to 2^V \times 2^V$ assigns each $m$ to its typed incidence $(\mathrm{in}(m), \mathrm{out}(m))$.
- $F = \{f_m\}_{m \in E}$ where each $f_m: \mathrm{Dom}(\mathrm{in}(m)) \times \mathrm{Dom}(u_m) \to \mathrm{Dom}(\mathrm{out}(m))$ is a deterministic *joint* structural function.
- $P$ is a product distribution over $U \cup U^{\mathrm{exo}}$.
- $P_0 = \{P_0(v)\}_{v \in V}$ is a *fallback exogenous distribution* for each variable, used when interventions delete all of $v$'s producing mechanisms (see §6).

**Reduction to Pearl.** A Pearl SCM is exactly the special case $|\mathrm{out}(m)| = 1$ for all $m \in E$, with $P_0$ trivial (never invoked).

---

## 3. The mechanism dependency graph

Define the directed graph $G_E$ with node set $E$ and edges $m_1 \to m_2$ whenever $\mathrm{out}(m_1) \cap \mathrm{in}(m_2) \neq \emptyset$.

**C1** requires $G_E$ to be acyclic. This induces a (not necessarily unique) topological order $m_{(1)}, \ldots, m_{(k)}$.

Note: mechanism-level acyclicity does *not* imply node-level acyclicity in any weaker sense — there is no node-level graph in this framework, only the bipartite blowup (§5) and the mechanism graph $G_E$.

---

## 4. Observational semantics

Sampling from $\mathcal{M}$:

1. For each $v \in V^{\mathrm{exo}}$: sample $u^{\mathrm{exo}}_v \sim P$, set $v := u^{\mathrm{exo}}_v$.
2. For each $m \in E$: sample $u_m \sim P$.
3. For each $m$ in topological order of $G_E$: compute $\mathrm{out}(m) := f_m(\mathrm{in}(m), u_m)$. (Joint assignment to all output variables of $m$.)

This induces the observational distribution $P^{\mathcal{M}}(V)$.

---

## 5. The bipartite blowup

The **bipartite blowup** $B(\mathcal{M})$ is the directed graph with node set $V \cup E$ and edge set

$$
\{(v, m) : v \in \mathrm{in}(m)\} \;\cup\; \{(m, v) : v \in \mathrm{out}(m)\}.
$$

$B(\mathcal{M})$ is bipartite by construction; alternation of variable and mechanism nodes is enforced by typed incidence.

**Proposition 5.1 (sampling equivalence).** Treat each mechanism node $m$ in $B(\mathcal{M})$ as a deterministic relay computing $f_m$ on its parents and noise $u_m$. Then the marginal of standard Pearl-style ancestral sampling on $B(\mathcal{M})$ over $V$ equals $P^{\mathcal{M}}(V)$.

*Proof.* Identical sampling procedure, identical functional dependencies. Variable nodes and mechanism nodes alternate in $B(\mathcal{M})$, so ancestral sampling on $B(\mathcal{M})$ visits each $f_m$ in a topological order consistent with $G_E$. □

This proposition is the bridge that lets us re-use Pearl machinery on a Hypergraph SCM — *for variable interventions only*. Mechanism interventions (§6) are not Pearl-expressible as single interventions.

---

## 6. Three do-operators

For $\mathcal{M} = (V, E, \rho, F, P, P_0)$:

### 6.1 Variable intervention

$\mathrm{do}(v = x)$ produces $\mathcal{M}^{v=x}$ by:
- Removing every $m$ with $v \in \mathrm{out}(m)$ from $E$.
- Adding $v$ to $V^{\mathrm{exo}}$ with point mass $\delta_x$.

Equivalent to standard Pearl intervention on $B(\mathcal{M})$.

### 6.2 Mechanism deletion (new)

$\mathrm{do}(\neg m)$ produces $\mathcal{M}^{\neg m}$ by:
- Removing $m$ from $E$.
- For each $v \in \mathrm{out}(m)$ that has no remaining producer: $v$ enters $V^{\mathrm{exo}}$ with distribution $P_0(v)$.

This is the operation Pearl cannot express as a single intervention. Its closest Pearl translation is the *simultaneous* multi-variable intervention $\mathrm{do}(v_1 = P_0, \ldots, v_p = P_0)$ for all $v_i \in \mathrm{out}(m)$ — which is multiple interventions, not one. The distinction matters experimentally: a single drug ablating one enzyme is one experiment, whereas Pearl's translation requires fixing every product simultaneously.

### 6.3 Mechanism replacement (new)

$\mathrm{do}(m \to m')$ produces $\mathcal{M}^{m \to m'}$ by replacing $f_m$ with $f_{m'}$, where $\rho(m') = \rho(m)$ (same incidence, different functional form).

Models drug substitution, enzyme variant swap, regulatory program edit.

---

## 7. The intervention space

The intervention space of $\mathcal{M}$ is

$$
\mathcal{I}(\mathcal{M}) \;=\; \underbrace{V \times \mathrm{Dom}(V)}_{\text{Pearl}} \;\cup\; \underbrace{\{\neg m : m \in E\}}_{\text{deletions}} \;\cup\; \underbrace{\{m \to m' : \rho(m') = \rho(m)\}}_{\text{replacements}}.
$$

This is strictly larger than Pearl's intervention space whenever $E$ contains any mechanism with $|\mathrm{out}(m)| \geq 2$ (otherwise mechanism-deletion reduces to a single variable intervention).

The slogan "mechanisms are first-class causal objects" is precisely the assertion that $\mathcal{I}(\mathcal{M}) \supsetneq V \times \mathrm{Dom}(V)$.

---

## 8. Counterfactual semantics (Pearl's three steps, generalized)

Given observed evidence $V = v$:

1. **Abduction.** For each $m \in E$, infer $u_m^* \in \mathrm{Dom}(u_m)$ such that $f_m(\mathrm{in}(m) = v|_{\mathrm{in}(m)},\, u_m^*) = v|_{\mathrm{out}(m)}$. For $v \in V^{\mathrm{exo}}$, $u^{\mathrm{exo},*}_v = v$.

2. **Action.** Apply intervention $\iota \in \mathcal{I}(\mathcal{M})$ to obtain $\mathcal{M}^*$.

3. **Prediction.** Re-evaluate $\mathcal{M}^*$ with abducted noise $\{u_m^*\} \cup \{u^{\mathrm{exo},*}_v\}$, sampling fresh noise only for variables newly added to $V^{\mathrm{exo}}$ by the intervention.

### 8.1 Abduction well-posedness

For each mechanism $m$, abduction requires solving $f_m(x_{\mathrm{in}}, u_m) = x_{\mathrm{out}}$ for $u_m$ given observed $x_{\mathrm{in}}, x_{\mathrm{out}}$. Three cases:

- **Determined.** $|\mathrm{out}(m)| = \dim(u_m)$ and $f_m(x_{\mathrm{in}}, \cdot)$ is invertible. Counterfactuals are unique.
- **Over-determined.** $|\mathrm{out}(m)| > \dim(u_m)$. The system is consistent only on a measure-zero subset of $\mathrm{Dom}(\mathrm{out}(m))$; the structural form *enforces* an output constraint (e.g., $C = D$ in the worked example). On-manifold observations admit unique abduction.
- **Under-determined.** $|\mathrm{out}(m)| < \dim(u_m)$. Abduction yields a noise distribution rather than a point; counterfactuals are non-unique without additional posterior structure on $u_m$.

The over-determined case is the *interesting* one — it is the formal home of conservation laws, stoichiometric coupling, and irreducible joint output structure. The minimal example exhibits this case.

---

## 9. Reduction to Pearl: what survives, what doesn't

**Survives** (via the bipartite blowup):
- Sampling semantics.
- Variable interventions.
- d-separation (target theorem §10).
- Counterfactuals when each $f_m$ is invertible-in-noise.

**Does not survive** as a single intervention in Pearl:
- Mechanism deletion $\mathrm{do}(\neg m)$ for $|\mathrm{out}(m)| \geq 2$.
- Mechanism replacement $\mathrm{do}(m \to m')$.

**Strictly larger:**
- The intervention space.
- The class of identifiable counterfactual queries (some queries unavailable in Pearl become available once the mechanism is named).

---

## 10. Target theorem: bipartite-blowup d-separation

**Theorem (T1, Soundness + Completeness).** Let $Z^* = Z \cup \mathrm{Det}_{\mathcal{M}}(Z)$ be the functional-determination closure. For pairwise disjoint $X, Y, Z \subseteq V$,

$$
X \perp_{P^{\mathcal{M}}} Y \mid Z \;\iff\; X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z
$$

where $d^*$-separation in $B(\mathcal{M})$ is standard Pearl d-separation with conditioning set $Z^*$. The completeness direction requires the standard faithfulness assumption.

**Proof.** See `THEOREM_T1.md`. The proof reduces T1 to Verma-Pearl 1990 and Geiger-Verma-Pearl 1990 via the explicit noise-augmented DAG $\tilde{B}(\mathcal{M})$, then translates determination closure between $\tilde{B}$ and $B$.

**Implementation.** A reference d*-separation algorithm is provided in `minimal_model/dseparation.py`; T1's predictions on the worked example are verified by `minimal_model/test_dseparation.py` (12 tests passing, including the deterministic-augmentation case $A \perp D \mid C$).

The reference implementation computes $\mathrm{Det}_{\mathcal{M}}(Z)$ via the **output-equality subcase** of functional determination: each mechanism may declare `output_equalities` — tuples of outputs that are structurally equal under $f_m$ — and the closure propagates equality through these declared groups. This handles the worked example (where $C \equiv D$ via $m_1$) and any joint mechanism whose joint output is supported on a coordinate-equality manifold. The fully general functional-determination closure of §10 — covering, e.g., a mechanism whose output is a deterministic function of its inputs alone, with no noise dependence — is a strict superset of what `dseparation.py` infers automatically; on such cases the user must declare the equality explicitly. This is sufficient for v1's scope and a documented v2 extension point.

---

## 11. Catalogued alternatives (post-v1 work)

### 11.1 Cyclic mechanism graphs

Allowing $G_E$ to contain cycles (reversible reactions, feedback regulation) requires fixed-point or measure-theoretic semantics. Bongers et al. 2018 ("Foundations of structural causal models with cycles and latent variables") provides the Pearl-side machinery; lifting to hypergraphs is straightforward in form, demanding in execution.

### 11.2 Markov-kernel mechanisms

Replace each $f_m$ with a Markov kernel $K_m: \mathrm{Dom}(\mathrm{in}(m)) \to \Delta(\mathrm{Dom}(\mathrm{out}(m)))$. Subsumes deterministic-with-noise but breaks abduction's noise-recovery formulation; counterfactuals require a different formal account (see Markov-category causality, Fritz 2020; Jacobs-Kissinger-Zanasi 2019).

### 11.3 Richer role typing

Generalize $\rho: E \to 2^V \times 2^V$ to a role assignment $\rho: E \to V^*$ with named roles (substrate, enzyme, product, modulator). Requires equivariance specifications (which role permutations leave $f_m$ invariant). Higher domain fidelity, modest formal cost.

### 11.4 Hyper-confounders

Latent common causes that influence multiple variables through one unobserved joint mechanism. Generalize ADMGs (acyclic directed mixed graphs) to acyclic directed mixed *hyper*graphs with bidirected hyperedges. Identification theory open.

### 11.5 Mechanism-level identifiability

State and prove an analogue of Pearl's do-calculus for $\mathrm{do}(\neg m)$ and $\mathrm{do}(m \to m')$ from observational data. Almost certainly requires additional structural assumptions beyond standard identifiability.

### 11.6 Causal abstraction as quotient

Beckers-Halpern (2019) abstraction conditions, expressed natively as collapsing subgraphs of $B(\mathcal{M})$ into single mechanism nodes. The hypergraph framework is a natural home for this since the resulting object remains a Hypergraph SCM (whereas in Pearl the abstraction sits outside the formalism).

---

## 12. Notational summary

| Symbol | Meaning |
|--------|---------|
| $V$ | set of variables |
| $E$ | set of mechanisms (hyperedges) |
| $m \in E$ | a single mechanism |
| $\mathrm{in}(m), \mathrm{out}(m)$ | input and output node sets of $m$ |
| $\rho$ | typed incidence assignment |
| $f_m$ | joint structural function of $m$ |
| $u_m$ | per-mechanism exogenous noise |
| $V^{\mathrm{exo}}$ | variables with no producer |
| $P_0(v)$ | fallback exogenous distribution for $v$ |
| $G_E$ | mechanism dependency graph |
| $B(\mathcal{M})$ | bipartite blowup |
| $\mathcal{I}(\mathcal{M})$ | intervention space |
| $M$ | typed incidence matrix |
