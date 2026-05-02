# H1+: Identifiability of Mechanism Deletion in HADMGs with Hidden Variables

This document extends T4 (`THEOREM_T4_T5.md`) to the case where some variables are unobserved. The headline result is **Theorem T6**: T4's truncated factorization formula remains valid in the presence of hidden variables, *provided* the target mechanism's boundary is fully observed. The boundary-violating case reduces to a Pearl ADMG identification problem on the bipartite blowup (**Theorem T7**), where the **hyper-hedge** — defined precisely below — is the structural obstruction. Completeness of the hyper-hedge criterion is stated as the principal remaining conjecture.

---

## 0. The setting

A **HADMG with hidden variables** is a Hypergraph SCM $\mathcal{M} = (V, E, \rho, F, P, P_0)$ together with a partition

$$
V = V^{\mathrm{obs}} \;\sqcup\; V^{\mathrm{lat}}
$$

into observed and hidden variables, and a partition

$$
E = E^{\mathrm{obs}} \;\sqcup\; E^{\mathrm{lat}}
$$

into observed and latent mechanisms, as in T4. v1 conventions C1–C4 hold throughout. The empirical access is $P(V^{\mathrm{obs}})$ — the marginal of $P^{\mathcal{M}}(V)$ over $V^{\mathrm{lat}}$.

T4 was the special case $V^{\mathrm{lat}} = \emptyset$; here we allow $V^{\mathrm{lat}} \neq \emptyset$.

---

## 1. The boundary condition

For a mechanism $m^\star \in E$, define its **boundary**:

$$
\partial m^\star \;:=\; \mathrm{in}(m^\star) \cup \mathrm{out}(m^\star) \;\subseteq\; V.
$$

We say $m^\star$ has an **observed boundary** if $\partial m^\star \subseteq V^{\mathrm{obs}}$. This is the structural property that determines whether T4's formula transfers directly.

---

## 2. Theorem T6: identifiability under observed boundary

**Theorem T6.** Let $\mathcal{M}$ be a HADMG with hidden variables. Let $m^\star \in E$ satisfy the observed-boundary condition $\partial m^\star \subseteq V^{\mathrm{obs}}$. Then $\mathrm{do}(\neg m^\star)$ is identifiable from $P(V^{\mathrm{obs}})$ via:

$$
P^{\mathcal{M}^{\neg m^\star}}(V^{\mathrm{obs}}) \;=\; \frac{P(V^{\mathrm{obs}})}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v).
$$

The mechanism factor $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is read off as a conditional in $P(V^{\mathrm{obs}})$.

### Proof

Two facts compose:

**Fact T6a.** Under v1 conventions, the conditional $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ in the *full-joint* distribution $P^{\mathcal{M}}(V)$ equals the *causal mechanism factor* — the pushforward of $P(u_{m^\star})$ through $f_{m^\star}(\mathrm{in}(m^\star), \cdot)$.

*Proof.* Noise independence: $u_{m^\star} \perp (V \setminus \mathrm{out}(m^\star))$. So for any conditioning set $Z \subseteq V \setminus \mathrm{out}(m^\star)$ with $\mathrm{in}(m^\star) \subseteq Z$:

$$
P(\mathrm{out}(m^\star) \mid Z) = P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star), Z \setminus \mathrm{in}(m^\star)) = P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)).
$$

The first equality uses $\mathrm{in}(m^\star) \subseteq Z$; the second uses noise independence and the fact that $\mathrm{out}(m^\star)$ is a deterministic function of $\mathrm{in}(m^\star)$ and $u_{m^\star}$. □

**Fact T6b.** Under the observed-boundary condition, the marginal conditional $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ in $P(V^{\mathrm{obs}})$ equals the corresponding full-joint conditional.

*Proof.* Since $\mathrm{out}(m^\star), \mathrm{in}(m^\star) \subseteq V^{\mathrm{obs}}$, the conditional $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is the same object computed from either $P(V)$ or $P(V^{\mathrm{obs}})$ — marginalizing out $V^{\mathrm{lat}}$ commutes with the conditioning (by Fubini, since the conditional is bounded). □

**Combining.** Apply T2 in its full-joint form:

$$
P^{\mathcal{M}^{\neg m^\star}}(V) = \frac{P(V)}{P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v).
$$

By Fact T6a, the denominator is well-defined and equals the mechanism factor. By Fact T6b, the denominator does not depend on $V^{\mathrm{lat}}$ — it can be pulled outside the integral when marginalizing:

$$
P^{\mathcal{M}^{\neg m^\star}}(V^{\mathrm{obs}}) = \int P^{\mathcal{M}^{\neg m^\star}}(V) \, dV^{\mathrm{lat}} = \frac{1}{P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v) \cdot \int P(V) \, dV^{\mathrm{lat}}.
$$

The last integral is $P(V^{\mathrm{obs}})$. Substituting yields T6. □

### What T6 says

T4's "always identifiable" claim was phrased for $V^{\mathrm{lat}} = \emptyset$. T6 strengthens it: the formula remains valid even when there are arbitrary hidden variables and arbitrary latent mechanisms, **as long as we don't try to delete a mechanism whose boundary touches the hidden subset**. The hidden structure is preserved in the surviving factors of $P(V^{\mathrm{obs}})$ and propagates correctly to the post-intervention observed marginal.

This is a stronger claim than it might appear. Pearl's variable-intervention identifiability requires hedge analysis even when the target $X$ is observed, because back-door paths through latents may obstruct adjustment. T6 says: in the hypergraph framework, observed-boundary mechanism deletion has *no analogous obstruction*. The chain-rule factorization at the mechanism level isolates the relevant factor cleanly.

---

## 3. Theorem T7: reduction in the boundary-violating case

When $\partial m^\star \not\subseteq V^{\mathrm{obs}}$, T6 does not apply. The full identifiability question reduces to a Pearl ADMG identification problem.

**Theorem T7.** Let $\mathcal{M}$ be a HADMG, $m^\star \in E$ with $\partial m^\star \not\subseteq V^{\mathrm{obs}}$. Then $\mathrm{do}(\neg m^\star)$ is identifiable from $P(V^{\mathrm{obs}})$ if and only if the conditional distribution

$$
P\!\left(\mathrm{out}(m^\star) \cap V^{\mathrm{obs}} \mid \mathrm{in}(m^\star) \cap V^{\mathrm{obs}}, \, \mathrm{do}(V^{\mathrm{obs}} \setminus \partial m^\star)\right)
$$

— or some closed sufficient sub-query — is identifiable in the bipartite-blowup ADMG $B^\dagger(\mathcal{M})$ via Shpitser-Pearl 2006's ID algorithm.

*Proof sketch.* Mechanism deletion in $\mathcal{M}$ corresponds, under the bipartite blowup, to a *stochastic intervention* on $\mathrm{out}(m^\star)$ in $B^\dagger(\mathcal{M})$ — assigning each output its $P_0$ distribution. Stochastic interventions reduce to standard $\mathrm{do}$-interventions for identifiability purposes (Pearl 2009, Bareinboim-Pearl 2016). The post-intervention observed marginal $P^{\neg m^\star}(V^{\mathrm{obs}})$ is identifiable in $\mathcal{M}$ iff the corresponding stochastic-intervention marginal is identifiable in $B^\dagger(\mathcal{M})$. By the soundness and completeness of Shpitser-Pearl's ID algorithm for Pearl ADMGs, this reduces to a hedge search on $B^\dagger(\mathcal{M})$. □

T7 is a *reduction*, not a new identifiability theorem. It says the boundary-violating case contains nothing new in principle — Pearl's existing machinery, applied to the right Pearl ADMG, settles the question.

---

## 4. Hyper-hedges

Define the **hyper-hedge** as the structural obstruction whose presence in $\mathcal{M}$ is equivalent to the existence of a Pearl hedge in $B^\dagger(\mathcal{M})$ for the query of T7.

### 4.1 Construction of $B^\dagger(\mathcal{M})$ for hidden-variable HADMGs

Extending the bipartite blowup of `THEOREM_T1.md` §5 to handle hidden variables:

- **Variable nodes.** All of $V = V^{\mathrm{obs}} \cup V^{\mathrm{lat}}$ are nodes in $B^\dagger$. Hidden variables are marked as such (Pearl ADMG-style: hidden variables become bidirected-edge endpoints after standard latent projection).
- **Mechanism nodes.** Each $m \in E$ becomes a node in $B^\dagger$. Observed mechanisms are deterministic given parents + noise. Latent mechanisms have hidden noise.
- **Edges.** $v \to m$ for $v \in \mathrm{in}(m)$, $m \to v$ for $v \in \mathrm{out}(m)$.
- **Latent projection.** Apply standard ADMG latent projection (Pearl 2009 §3.7): marginalize out $V^{\mathrm{lat}}$ and replace each path through a hidden variable by appropriate directed and bidirected edges among observed nodes.

### 4.2 Definition of hyper-hedge

A **hyper-hedge for $m^\star$** in $\mathcal{M}$ is a pair of sub-hypergraphs $(\mathcal{F}, \mathcal{F}')$ of $B^\dagger(\mathcal{M})$ such that:

1. **C-forest analog.** Each is a connected sub-hypergraph rooted at the same mechanism node $r$, and is contained in a single hyper-c-component (defined below).
2. **Strict containment.** $\mathcal{F}' \subsetneq \mathcal{F}$.
3. **Intervention obstruction.** The variables of $\mathcal{F}'$ — specifically, $\mathrm{out}(m^\star) \cap \mathcal{F}'$ — lie in the do-set of T7's stochastic intervention; the variables of $\mathcal{F} \setminus \mathcal{F}'$ contain a target observation.

**Hyper-c-component.** A maximal subset of $V^{\mathrm{obs}}$ connected by latent-mechanism hyperedges in $B^\dagger(\mathcal{M})$, where two observed variables share a latent-mechanism hyperedge iff they are both outputs (or both confounded outputs after latent projection) of some latent mechanism or hidden ancestor.

When $|\mathrm{out}(m)| = 1$ for all $m$, hyper-c-components reduce to standard Pearl c-components and hyper-hedges reduce to Pearl hedges (Shpitser-Pearl 2006, Definition 5).

### 4.3 Conjecture (Hyper-hedge completeness)

**Conjecture H1+ (full).** Let $\mathcal{M}$ be a HADMG with hidden variables. Then $\mathrm{do}(\neg m^\star)$ is identifiable from $P(V^{\mathrm{obs}})$ if and only if there is no hyper-hedge for $m^\star$ in $B^\dagger(\mathcal{M})$.

**Status.** The forward direction (no hyper-hedge ⇒ identifiable) follows from T6 + T7 + Shpitser-Pearl 2006's soundness, plus a careful argument that the bipartite blowup preserves the relevant hedge structure. The reverse direction (hyper-hedge ⇒ unidentifiable) lifts Shpitser-Pearl's completeness via the same blowup; correctness depends on a technical lemma about deterministic-relations preservation under hedge counterexample constructions.

We state H1+ (full) as the principal open conjecture of v1. The pieces are in place; full proof is a substantial undertaking on the order of Shpitser-Pearl 2006 itself, and we do not attempt it here.

---

## 5. What v1 commits to

| Result | Statement | Status |
|---|---|---|
| **T6** | T4 extends to hidden-variable HADMGs under the observed-boundary condition. | **Proved.** |
| **T7** | Boundary-violating case reduces to Pearl ADMG identification on $B^\dagger(\mathcal{M})$. | **Proof sketched** via reduction to Shpitser-Pearl 2006. |
| **Hyper-hedge** | Definition; equivalence to Pearl hedge under bipartite blowup. | **Defined; structural equivalence stated** without full proof. |
| **H1+ (full)** | Hyper-hedge completeness: no hyper-hedge ⇔ identifiable. | **Open conjecture.** |

What this means honestly: T6 is a clean theorem with a clean proof. T7 is a reduction that says "there's nothing new to prove here in principle — apply Pearl's machinery." Hyper-hedge is a definitional bridge that makes T7 concrete. H1+ (full) is the completeness statement that would close the theory; we have not proved it, and we mark it open.

This is the boundary of v1. Beyond it lies a substantive paper-length effort — proving H1+ completeness requires lifting a delicate deterministic-relations argument from Shpitser-Pearl, which is itself a 30-page technical proof. The right next step, if pursued, is a focused effort dedicated to that single completeness theorem, ideally with a coauthor experienced in Pearl-tradition causal inference.

---

## 6. Worked instance: a hidden-variable HADMG

Extend the reaction network with a hidden variable $W$ and a latent mechanism $m_W: \emptyset \to W$. Let $W$ be an input to a *new* observed mechanism $m_3: C, W \to G$, where $G$ is observed.

- $V^{\mathrm{obs}} = \{A, B, C, D, E, F\}$
- $V^{\mathrm{lat}} = \{W\}$ (new variable, hidden)
- $E = \{m_W, m_1, m_2\}$ where $m_W: \emptyset \to W$ (latent mechanism producing hidden var)
- $m_1: A, B \to C, D$ (observed; boundary observed)
- $m_2: C, E, W \to F$ (observed; boundary contains hidden $W$)

**Verdict for $m_1$.** $\partial m_1 = \{A, B, C, D\} \subseteq V^{\mathrm{obs}}$. T6 applies. $\mathrm{do}(\neg m_1)$ identifiable; formula:

$$
P^{\neg m_1}(V^{\mathrm{obs}}) = \frac{P(V^{\mathrm{obs}})}{P(C, D \mid A, B)} \cdot \delta_0(C) \delta_0(D).
$$

The hidden $W$ enters $P(V^{\mathrm{obs}})$ through its propagation via $m_2$ to $F$, but does not enter the formula's surface — it's preserved in $P(V^{\mathrm{obs}})$ and propagates correctly through the post-intervention marginal.

**Verdict for $m_2$.** $\partial m_2 = \{C, E, W, F\}$, with $W \in V^{\mathrm{lat}}$. T6 fails. T7 applies: the question reduces to whether $P(F \mid \mathrm{do}(C, E)) $ — the variable-intervention marginal in $B^\dagger(\mathcal{M})$ — is identifiable. In this graph, the latent mechanism $m_W$ produces only $W$ (no observed sibling), so the latent has no back-door path to $F$ through observed variables. The Pearl identification succeeds; $\mathrm{do}(\neg m_2)$ is identifiable, even though T6 does not detect it.

**Verdict for $m_W$.** $\partial m_W = \{W\}$ entirely hidden. T6 fails. T7: the post-intervention marginal $P^{\neg m_W}(V^{\mathrm{obs}})$ corresponds to a stochastic intervention on $W$ in $B^\dagger(\mathcal{M})$, which propagates through $m_2$ to $F$. Identifiability of this query depends on whether $W$'s effect on $F$ can be disentangled from confounding paths — a standard Pearl ID problem. In this specific construction (no other observed children of $W$), the answer is yes: $W$'s marginal effect is entirely captured by its propagation through $m_2$, and $\mathrm{do}(\neg m_W)$ has the closed form $P^{\neg m_W}(V^{\mathrm{obs}}) = P(A, B, C, D, E) \cdot P(F \mid C, E, W = 0)$. The numerical version is verified in `test_hadmg.py::test_T7_recovers_identifiable_boundary_violating_case`.

The verdicts illustrate the v1 status:

| Mechanism | T6 verdict | True identifiability | Gap |
|---|---|---|---|
| $m_1$ | Identifiable (boundary observed) | Identifiable | T6 detects. |
| $m_2$ | Refuses (boundary fails) | Identifiable via T7 | T6 conservatively refuses; T7 reduction succeeds. |
| $m_W$ | Refuses (boundary fails) | Identifiable via T7 | Same. |

The implementation `is_mechanism_deletion_identifiable(...)` checks T6 (sufficient condition). Cases where the verdict is "no" but identifiability holds via T7 are flagged for future work — they correspond to the open hyper-hedge completeness conjecture.

---

## 7. Summary

H1+ closes the H1 program with three deliverables:

1. **T6** — a clean sufficient condition (observed boundary) under which T4's formula transfers verbatim to hidden-variable HADMGs.
2. **T7** — a reduction of the boundary-violating case to Pearl ADMG identification, via the bipartite blowup.
3. **Hyper-hedge** — a structural definition that makes T7 concrete and connects to the existing Pearl hedge machinery.

The remaining open content (hyper-hedge completeness) is the natural research-grade extension. v1 has built the formalism and proved the easy half; the hard half is a focused future effort.

---

## References

- Shpitser, I. & Pearl, J. (2006). "Identification of conditional interventional distributions." *UAI 22*.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press, §3.7 (latent projection).
- Tian, J. & Pearl, J. (2002). "A general identification condition for causal effects." *AAAI 18*.
- Bareinboim, E. & Pearl, J. (2016). "Causal inference and the data-fusion problem." *PNAS* 113.
