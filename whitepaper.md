# Causal Hypergraph SCMs: A First-Class Calculus for Mechanism-Level Intervention

---

## Abstract

We develop a strict generalization of Pearl's structural causal models in which mechanisms — typed hyperedges with multiple inputs and multiple jointly-produced outputs — are first-class causal objects. The framework is motivated by domains where Pearl's three structural assumptions (single-output equations, dyadic edges, node-level DAG) fail to faithfully capture the underlying causal structure: chemical reactions with stoichiometric coupling, regulatory pathways, n-ary knowledge-base relations, and multi-target therapeutic interventions.

The contribution is not new expressive power — Pearl with sufficient latents is universal — but **first-class addressability** of mechanisms, which strictly extends the intervention vocabulary and yields closed-form identifiers where Pearl's translation requires multi-variable case analysis. We prove three foundational results: a graphical Markov property via a bipartite-blowup construction (T1), a mechanism-level chain rule yielding identifiability of mechanism deletion and replacement under causal sufficiency (T2, T3), and an asymmetry between mechanism-level and variable-level interventions in the presence of hidden mechanisms (T4) — mechanism deletion admits a single closed-form identifier under standard noise-independence assumptions, requiring no Pearl-style hedge analysis. We extend this to hidden-variable HADMGs under a clean boundary condition (T6), and reduce the boundary-violating case to standard Pearl ADMG identification on the bipartite blowup (T7), where genuine unidentifiability via a hyper-hedge becomes possible.

The principal open conjecture is the completeness of a hyper-hedge characterization for boundary-violating mechanism deletions. A reference implementation reproduces every numerical claim to within sampling tolerance and is available in the supplementary material.

---

## 1. Introduction

### 1.1 Pearl's hidden assumptions

Pearl's structural causal model (SCM) framework (Pearl 2009) rests on three structural assumptions that are easy to miss because they are baked into the notation:

1. **Single-output equations.** Each structural equation $v := f_v(\mathrm{pa}(v), u_v)$ produces exactly one variable on the left-hand side.
2. **Dyadic edges.** Causal influence is encoded by directed edges between pairs of variables.
3. **Node-level acyclicity.** The induced graph is a DAG over the variable set.

These assumptions are extraordinarily productive. They support do-calculus, identifiability theory, counterfactual computation via twin networks, and a polynomial-time decision procedure for conditional independence. Most of modern causal inference rests on them.

But they fail in a class of domains where the relevant causal mechanisms are not naturally decomposable into pairwise relations with single outputs. A chemical reaction $A + B \to C + D$ produces $C$ and $D$ jointly, with rates linked by stoichiometry; the structural identity $\Delta C = \Delta D$ is a feature of the mechanism, not an emergent property of independent edges. A protein complex of five subunits binding jointly is not the same as ten pairwise PPIs. An n-ary knowledge-base tuple `(person, company, role, start, end)` is genuinely 5-ary; binarizing destroys the joint constraints. A drug ablating a single enzyme acts on a *reaction* (a hyperedge), not on a *molecule* (a node).

In each case, the natural causal primitive is not a structural equation with one output but a **mechanism**: a typed hyperedge with multiple inputs and multiple jointly-produced outputs, governed by a single shared noise process. Encoding such systems in Pearl requires either (a) ignoring the joint structure and accepting the resulting unfaithfulness, or (b) introducing latent variables for each shared random component, at the cost of representational bloat and an ill-fit between the formal latents and the substantive mechanisms.

### 1.2 Contribution

We develop the formal alternative: hypergraph SCMs in which mechanisms are first-class causal objects. The contribution is not new expressive power — Pearl with sufficient latents matches any joint distribution and any interventional query — but **first-class addressability**. Naming mechanisms as primary objects of the formalism unlocks new intervention operations ($\mathrm{do}(\neg m)$ for mechanism deletion, $\mathrm{do}(m \to m')$ for replacement) and yields a substantively different identifiability theory.

Three results form the technical core:

- **A graphical Markov property** (Theorem T1). A bipartite-blowup construction lifts Pearl's d-separation, with a deterministic-relations augmentation handling joint mechanism outputs. This gives a polynomial-time conditional-independence oracle on hypergraph SCMs, sound and complete under standard faithfulness.

- **A mechanism-level identifiability calculus** (Lemma 1.1, Theorems T2/T3). Mechanism interventions admit truncated factorization formulas: $P(V \mid \mathrm{do}(\neg m^\star)) = P(V) / P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)) \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v)$ under causal sufficiency. The mechanism factor $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ — the joint conditional of a mechanism's outputs given its inputs — generalizes Pearl's parental factor.

- **An asymmetry between mechanism and variable interventions** (Theorem T4). In a hypergraph ADMG (HADMG) with all variables observed, mechanism deletion admits a closed-form identifier from observational data, even in the presence of hidden mechanisms. By contrast, variable interventions reduce (Theorem T5) to Pearl multi-variable ID on the bipartite blowup — case-analytic rather than uniform. Under hidden *variables* (§7) the asymmetry sharpens further: T6 closes the observed-boundary case in closed form, and only T7's boundary-violating case can genuinely fail via a hyper-hedge.

Theorem T6 extends T4 to hidden-variable HADMGs under a clean boundary condition. Theorem T7 reduces the boundary-violating case to standard Pearl ADMG identification. The completeness of a corresponding hyper-hedge characterization is stated as the principal remaining open conjecture.

### 1.3 Scope

We commit explicitly to four conventions in v1: mechanism-level acyclicity (C1), deterministic structural functions with explicit per-mechanism noise (C2), bipartite role typing — each hyperedge has only an input/output partition (C3), and a single producer per variable (C4). Cyclic hypergraph SCMs, Markov-kernel mechanisms, richer role typing (substrate / enzyme / product), and mechanism-correlated noise are flagged as future extensions; the relevant prior work is reviewed in §3.

A reference implementation in Python (~800 lines, NumPy only) reproduces every numerical claim in this paper to within sampling tolerance. Five test suites comprising 60 tests verify structural sanity, all three intervention regimes, T1's d*-separation predictions, T2/T3's truncated factorizations, T4's asymmetry under latent confounding, and T6's behavior with a hidden variable.

### 1.4 Roadmap

§2 reviews related work. §3 develops the formalism and the bipartite blowup. §4 proves T1. §5 introduces the mechanism-level chain rule and the truncated factorization formulas (T2/T3). §6 establishes the central asymmetry (T4) and reduces the variable case to Pearl (T5). §7 extends to hidden variables (T6/T7, hyper-hedge). §8 consolidates the worked example. §9 catalogs open problems. §10 concludes.

---

## 2. Related Work

### 2.1 Pearl-tradition causal inference

The framework builds directly on Pearl's structural causal model machinery (Pearl 2009). The d-separation theorem (Verma & Pearl 1990; Geiger, Verma & Pearl 1990) is the foundation for our T1; the deterministic-relations augmentation we require for joint mechanism outputs lifts existing Pearl machinery (Geiger & Pearl 1990). Our identifiability results stand in close relation to Pearl's variable-intervention identifiability (Tian & Pearl 2002; Shpitser & Pearl 2006) — T5 is a direct reduction, while T4 establishes the asymmetry that gives our framework its distinctive epistemic profile.

The hedge-obstruction characterization of Shpitser-Pearl 2006 is the standard against which we compare. Our hyper-hedge (§7) is a structural lift of their hedge through the bipartite blowup; we conjecture a corresponding completeness theorem and outline what would be required to prove it.

### 2.2 Beyond Pearl: cyclic, kernel, and categorical extensions

Several lines of prior work have relaxed Pearl's structural assumptions. Bongers, Forré & Mooij (2018) develop SCM theory under cyclic structure; their fixed-point semantics will be relevant when we relax convention C1. Halpern's actual-causality framework (Halpern 2016) loosens the structural-equation form in ways that partially accommodate joint mechanisms but does not commit to a hypergraph formalism.

Categorical causality, particularly the Markov-category framework of Fritz (2020) and the string-diagrammatic causality of Jacobs, Kissinger & Zanasi (2019), provides what is arguably the most principled foundation for "many-to-many" causal mechanisms. Our hypergraph SCMs are, viewed through the categorical lens, generators in a free Markov category whose morphisms are mechanisms. A categorical reformulation is natural future work and would clarify several v1 design choices (especially the do-operator's semantics and the abduction step in counterfactuals).

### 2.3 Causal abstraction

The hypergraph framework is a natural setting for causal abstraction in the sense of Rubenstein et al. (2017) and Beckers & Halpern (2019). Their condition for collapsing a sub-graph into a single causal unit is precisely the condition under which a sub-mechanism becomes a hyperedge. Causal abstraction is an internal operation in our framework, where in Pearl it sits outside the formalism.

### 2.4 Hypergraph machine learning

A separate, large body of work addresses hypergraph-structured neural networks: AllSet and AllSetTransformer (Chien et al. 2022), HGNN+ (Gao et al. 2022), the Hypergraph Transformer (Kim et al. 2022), and — implicitly — AlphaFold 2's Evoformer (Jumper et al. 2021). These systems are *descriptive*: they learn representations of hypergraph data without committing to a causal interpretation. Our framework is the first, to our knowledge, to develop a causal theory of hypergraph SCMs. The connection to hypergraph ML is a direction for future work; a "causal hypergraph transformer" with attention biases respecting typed incidence and do-operators as structured masking would be a natural target.

### 2.5 Petri nets and chemical reaction networks

The hypergraph perspective on causality has been implicit in systems-biology modeling for decades. Petri nets (Murata 1989) are precisely many-to-many causal hypergraphs — transitions are hyperedges with input places and output places, and the firing semantics matches our mechanism-deletion intuition. Chemical reaction network theory (Feinberg 2019) develops a substantial literature on stoichiometric coupling, multi-output reactions, and conservation laws — the empirical content that motivates our framework. To our knowledge, this prior work has not been connected to formal causal-inference theory; our framework provides that bridge.

---

## 3. Formalism

### 3.1 Variables, mechanisms, typed incidence

Let $V = \{v_1, \ldots, v_n\}$ be a finite set of *variables*, and $E = \{m_1, \ldots, m_k\}$ a finite set of *mechanisms* (hyperedges). For each $m \in E$, let $\mathrm{in}(m), \mathrm{out}(m) \subseteq V$ be its input and output sets, with $\mathrm{in}(m) \cap \mathrm{out}(m) = \emptyset$. Write $V^{\mathrm{exo}} = \{v \in V : \nexists m, v \in \mathrm{out}(m)\}$ for variables produced by no mechanism.

The **typed incidence matrix** $M \in \{-1, 0, +1\}^{|V| \times |E|}$ is defined by $M[v, m] = +1$ if $v \in \mathrm{out}(m)$, $-1$ if $v \in \mathrm{in}(m)$, and $0$ otherwise. The standard hypergraph incidence matrix $|M|$ is recovered by absolute value; the *typing* (sign) is what distinguishes a causal hypergraph from a relational hypergraph.

### 3.2 Definition

A **Hypergraph SCM** is a tuple $\mathcal{M} = (V, E, \rho, F, P, P_0)$ where $\rho : E \to 2^V \times 2^V$ assigns each $m$ to its typed incidence $(\mathrm{in}(m), \mathrm{out}(m))$; $F = \{f_m\}_{m \in E}$ is a collection of *joint structural functions* $f_m : \mathrm{Dom}(\mathrm{in}(m)) \times \mathrm{Dom}(u_m) \to \mathrm{Dom}(\mathrm{out}(m))$; $P$ is a product distribution over per-mechanism and per-exogenous-variable noise terms; and $P_0 = \{P_0(v)\}_{v \in V}$ is a *fallback distribution* invoked when interventions delete a variable's producing mechanism.

A Pearl SCM is precisely the special case $|\mathrm{out}(m)| = 1$ for all $m$, with $P_0$ trivial.

### 3.3 v1 conventions

We commit to four conventions throughout v1; alternatives are catalogued in §9.

- **C1. Mechanism-level acyclicity.** The mechanism dependency graph $G_E$ — with edge $m_1 \to m_2$ whenever $\mathrm{out}(m_1) \cap \mathrm{in}(m_2) \neq \emptyset$ — is a DAG.
- **C2. Deterministic structural functions with independent noise.** Each $f_m$ is deterministic; stochasticity is carried entirely by independent per-mechanism noise $u_m$.
- **C3. Bipartite role typing.** Each hyperedge has only an input/output partition; richer roles (substrate, enzyme, etc.) are deferred.
- **C4. Single producer.** $\mathrm{out}(m_1) \cap \mathrm{out}(m_2) = \emptyset$ for distinct $m_1, m_2$ — the hypergraph analogue of Pearl's "one structural equation per variable."

### 3.4 Sampling

Sampling from $\mathcal{M}$: for each $v \in V^{\mathrm{exo}}$, draw $v \sim P_v^{\mathrm{exo}}$. For each $m \in E$, draw $u_m \sim P_m$. Then evaluate mechanisms in topological order of $G_E$: $\mathrm{out}(m) := f_m(\mathrm{in}(m), u_m)$. This induces the observational distribution $P^{\mathcal{M}}(V)$.

### 3.5 The bipartite blowup

The **bipartite blowup** $B(\mathcal{M})$ is the directed graph with node set $V \cup E$ and edges $v \to m$ for $v \in \mathrm{in}(m)$, $m \to v$ for $v \in \mathrm{out}(m)$. By C1, $B(\mathcal{M})$ is bipartite-acyclic.

**Proposition 3.1 (sampling equivalence).** Treat each mechanism node $m$ in $B(\mathcal{M})$ as a deterministic relay computing $f_m$ on its parents and noise $u_m$. Standard Pearl-style ancestral sampling on $B(\mathcal{M})$ marginalizes to $P^{\mathcal{M}}(V)$.

The proof is direct from the sampling procedures' identity (full proof: `THEOREM_T1.md` Lemma 2.2). The bipartite blowup is the bridge that lets us reuse Pearl machinery on hypergraph SCMs — but only for variable interventions; mechanism interventions are not Pearl-expressible as single operations (§5).

### 3.6 Three intervention operators

Three do-operators are well-defined on $\mathcal{M}$:

- **Variable intervention.** $\mathrm{do}(v = x)$: delete every $m$ with $v \in \mathrm{out}(m)$, set $v$ to point mass $\delta_x$ in $V^{\mathrm{exo}}$.
- **Mechanism deletion.** $\mathrm{do}(\neg m)$: delete $m$ from $E$. Variables in $\mathrm{out}(m)$ now lacking a producer fall back to $P_0$.
- **Mechanism replacement.** $\mathrm{do}(m \to m')$: replace $f_m$ with $f_{m'}$, where $\rho(m') = \rho(m)$.

The intervention space $\mathcal{I}(\mathcal{M}) = V \times \mathrm{Dom}(V) \cup \{\neg m : m \in E\} \cup \{m \to m' : \rho(m') = \rho(m)\}$ strictly contains Pearl's $V \times \mathrm{Dom}(V)$ whenever some $m$ has $|\mathrm{out}(m)| \geq 2$. This containment is the formal content of "first-class addressability of mechanisms."

---

## 4. Graphical reasoning: d*-separation (T1)

**Theorem T1 (Bipartite-blowup d*-separation).** Let $\mathrm{Det}_{\mathcal{M}}(Z)$ denote the *functional-determination closure* of $Z \subseteq V$ — the smallest superset of $Z$ closed under the relation "if all parents of $v$ in $B(\mathcal{M})$ are in the set, then so is $v$." Define $d^*$-separation in $B(\mathcal{M})$ as standard Pearl d-separation with conditioning set $Z^* := Z \cup \mathrm{Det}_{\mathcal{M}}(Z)$. Then for pairwise disjoint $X, Y, Z \subseteq V$:

$$
X \perp_{P^{\mathcal{M}}} Y \mid Z \;\iff\; X \perp^{d^*}_{B(\mathcal{M})} Y \mid Z
$$

(under standard faithfulness for the completeness direction).

**Proof sketch.** Construct the explicit noise-augmented DAG $\tilde{B}(\mathcal{M})$ over $V \cup E \cup U$, where each non-noise node is a deterministic function of its parents. Apply Verma-Pearl 1990's d-separation theorem with the deterministic-relations augmentation of Geiger-Verma-Pearl 1990. Translate back via the equality $\mathrm{Det}_{\tilde{B}}(Z) \cap V = \mathrm{Det}_{\mathcal{M}}(Z)$ and the fact that paths between non-noise nodes in $\tilde{B}$ and $B$ coincide. Full proof: `THEOREM_T1.md` §3–§4.

The deterministic augmentation is necessary, not cosmetic. The minimal example (§8) exhibits a structural identity $C \equiv D$ produced by a joint mechanism. Plain d-separation on $B(\mathcal{M})$ would predict $A \not\perp D \mid C$ (the path $A \to m_1 \to D$ traverses no node of $\{C\}$); but conditionally on $C = c$, the structural identity pins $D = c$, so $A \perp D \mid C$ in distribution. The augmentation $Z^* = Z \cup \mathrm{Det}(Z)$ adds $D$ to the conditioning set when $C$ functionally determines it, recovering the correct d-separation verdict.

**Corollaries.** $P^{\mathcal{M}}$ satisfies the local Markov property relative to $B(\mathcal{M})$ (each $v$ is independent of its non-descendants given its parents — the parent mechanism node and any deterministically-coupled siblings); $B(\mathcal{M})$ with $d^*$-augmentation is an I-map for $P^{\mathcal{M}}$ (perfect under faithfulness); variable interventions admit a truncated-factorization analog.

A reference implementation of d*-separation runs in $O(|V|^2 + |E|^2)$ time and is verified on twelve test cases including the deterministic-augmentation case above.

---

## 5. Mechanism interventions and the chain rule

### 5.1 Lemma 1.1: the mechanism-level chain rule

**Lemma 1.1.** Under v1 conventions C1–C4 and causal sufficiency,

$$
P^{\mathcal{M}}(V) = \prod_{v \in V^{\mathrm{exo}}} P(v) \cdot \prod_{m \in E} P\!\left(\mathrm{out}(m) \mid \mathrm{in}(m)\right).
$$

Each factor $P(\mathrm{out}(m) \mid \mathrm{in}(m))$ is the **mechanism factor** of $m$ — the conditional distribution of its joint output given its joint input, equivalent to the pushforward of $P(u_m)$ through $f_m(\mathrm{in}(m), \cdot)$.

The mechanism factor generalizes Pearl's parental factor $P(v \mid \mathrm{pa}(v))$ to *joint* conditional distributions over the outputs of a mechanism. C4 ensures these factors do not overlap. Independence of noise across mechanisms (C2) makes the factors conditionally independent given the topological-prefix evidence.

### 5.2 Theorem T2: truncated factorization for mechanism deletion

**Theorem T2.** Under v1 conventions and causal sufficiency, for any $m^\star \in E$:

$$
P^{\mathcal{M}^{\neg m^\star}}(V) = \frac{P^{\mathcal{M}}(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v).
$$

**Proof.** Apply Lemma 1.1 to both pre- and post-intervention distributions. Deletion removes the $m^\star$ factor from the product; under C4, every $v \in \mathrm{out}(m^\star)$ becomes orphaned and acquires a $P_0$ factor. Algebraic rearrangement yields the displayed formula. □

### 5.3 Theorem T3: mechanism replacement

**Theorem T3.** Under the same conditions, for any replacement $m'$ with $\rho(m') = \rho(m^\star)$:

$$
P^{\mathcal{M}^{m^\star \to m'}}(V) = \frac{P^{\mathcal{M}}(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot P_{f_{m'}}\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right).
$$

**Corollary T3.1.** Mechanism deletion is the special case of replacement where $m'$ is the *trivial mechanism* whose conditional distribution is the product of $P_0$ fallbacks, independent of inputs. Deletion and replacement are unified: deletion is "set this mechanism to its $P_0$-default" replacement.

### 5.4 What this gives, and why mechanism-level is *more* identifiable than Pearl multi-variable

T2 and T3 are clean truncated-factorization formulas, mathematically simple but substantively important. The Pearl analogue of $\mathrm{do}(\neg m^\star)$ is the *stochastic multi-variable* intervention $\mathrm{do}\!\left(\mathrm{out}(m^\star) \sim \prod P_0\right)$, which under Pearl ADMG ID reduces to a multi-variable ID problem with case-analytic output (Bareinboim-Pearl 2016). The hypergraph framework, by contrast, treats $\mathrm{do}(\neg m^\star)$ as a *single* operation and gives it a closed-form identifier read directly from Lemma 1.1.

This is the first technical observation that mechanism-level identification is structurally simpler than its Pearl translation: a single expression replaces an algorithmic search. It anticipates the closed-form-vs-case-analytic asymmetry of T4.

---

## 6. The asymmetry: identifiability under hidden mechanisms

### 6.1 Hypergraph ADMG (HADMG)

A **HADMG** is a Hypergraph SCM in which a subset $E^{\mathrm{lat}} \subseteq E$ of mechanisms is marked *latent*: their structural functions $f_m$ are unknown, though their typed incidence is known. Observed mechanisms $E^{\mathrm{obs}} = E \setminus E^{\mathrm{lat}}$ have known $f_m$. For v1, all variables are observed: $V = V^{\mathrm{obs}}$.

### 6.2 Theorem T4

**Theorem T4 (Mechanism-deletion identifiability under hidden mechanisms).** Let $\mathcal{M}$ be a HADMG with $V^{\mathrm{lat}} = \emptyset$. For any $m^\star \in E$ — observed or latent —

$$
P^{\mathcal{M}^{\neg m^\star}}(V) = \frac{P^{\mathcal{M}}(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v),
$$

and this is identifiable from $P(V)$.

**Proof.** Lemma 1.1's proof did not require observability of $f_m$ — only C1–C4 and noise independence. The mechanism factor $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is a conditional distribution in $P(V)$ — directly observable as a conditional in the data, regardless of whether $m^\star$ is latent. The rest of T2's proof goes through verbatim. □

### 6.3 The asymmetry, made precise

The asymmetry between T4 and T5 has two distinct components, which we separate carefully.

| Intervention type | Identifying expression under v1 + $V^{\mathrm{lat}} = \emptyset$ |
|---|---|
| $\mathrm{do}(\neg m)$ — mechanism deletion | **Closed form (T4):** $P(V) / P(\mathrm{out}(m) \mid \mathrm{in}(m)) \cdot \prod P_0$. Always identifiable; $O(\lvert V \rvert + \lvert E \rvert)$. |
| $\mathrm{do}(m \to m')$ — mechanism replacement | **Closed form (T3 + T4):** same denominator, replacement factor in numerator. |
| $\mathrm{do}(v = x)$ — variable intervention | **Reduction (T5):** Pearl multi-variable ID on $B^\dagger(\mathcal{M})$. |

**(a) Closed-form vs case-analytic.** Mechanism interventions admit a single explicit identifying formula readable directly from the chain rule. Variable interventions require running Pearl's ID algorithm on the bipartite blowup; the algorithm always terminates, but its output is case-analytic.

**(b) Robustness to hidden mechanisms.** T4's formula is invariant to whether $m^\star$ is observed or latent — only the typed incidence is needed. Variable-intervention identifiability via T5 depends on the specific bidirected-edge structure induced by latent mechanisms in $B^\dagger(\mathcal{M})$.

A natural further question is whether T5 ever returns *unidentifiable* on a HADMG satisfying v1 conventions. Under HSCM intervention semantics, $\mathrm{do}(v=x)$ deletes the mechanism producing $v$ and orphans its siblings to $P_0$; this corresponds in $B^\dagger(\mathcal{M})$ to a multi-variable intervention spanning the entire output set, which severs all bidirected edges that the producing latent mechanism would otherwise contribute. We have not been able to exhibit a v1 HADMG on which a variable intervention is genuinely unidentifiable, and we conjecture none exist — but state this as observation rather than theorem. The asymmetry's strongest form, where mechanism deletion succeeds and variable intervention genuinely fails, is the hidden-variable setting (T7) treated in §7.

### 6.4 The construction-vs-discovery objection

A skeptical reading: T4's positive result is a consequence of how mechanisms are defined to factorize cleanly in Lemma 1.1. The asymmetry is by construction, not by discovery.

The reply is that this is *exactly* the contribution. Naming mechanisms as first-class causal objects is the modeling choice; the chain-rule factorization is its consequence; the identifiability gain is the epistemic payoff — the closed-form identifier of T4 and the experimentally faithful intervention vocabulary. The point is not that the hypergraph framework identifies queries Pearl cannot identify (under v1 conventions, both formalisms reach the same answer), but that the hypergraph framework presents a single physical experiment as a single intervention object rather than a multi-variable translation.

The empirical content is non-trivial: a single drug ablation corresponds to one $\mathrm{do}(\neg m)$ in the hypergraph framework versus a multi-variable $\mathrm{do}$ in Pearl. The two frameworks agree on the resulting distribution but disagree on the *structure of the experiment*. The hypergraph framework's structure matches experimental reality.

### 6.5 Theorem T5: variable interventions reduce to Pearl

**Theorem T5.** Let $\mathcal{M}$ be a HADMG with $V^{\mathrm{lat}} = \emptyset$, and $X, Y \subseteq V$ disjoint. Then $P(Y \mid \mathrm{do}(X = x))$ is identifiable from $P(V)$ in $\mathcal{M}$ iff it is identifiable in the Pearl ADMG $B^\dagger(\mathcal{M})$ obtained from $B(\mathcal{M})$ by representing each latent mechanism $m^{\mathrm{lat}}$ as a single hidden confounder $u_{m^{\mathrm{lat}}}$ with bidirected edges among its outputs.

**Proof sketch.** The bipartite blowup, with latent-mechanism hidden noises represented as Pearl latents, is a Pearl ADMG. Sampling equivalence (Proposition 3.1) implies marginal-distribution agreement on $V$. Standard Shpitser-Pearl 2006 machinery applies. Full sketch: `THEOREM_T4_T5.md` §2.

T5 is a *reduction*, not a new theorem. The hypergraph framework requires no new identifiability theory for variable interventions — Pearl's theory applies via the bipartite blowup.

---

## 7. Hidden variables: T6, T7, and hyper-hedges

### 7.1 HADMGs with hidden variables

The general HADMG allows $V = V^{\mathrm{obs}} \sqcup V^{\mathrm{lat}}$: some variables are unobserved. The empirical access is the marginal $P(V^{\mathrm{obs}})$. T4's clean positive result was specific to $V^{\mathrm{lat}} = \emptyset$.

For a mechanism $m^\star$, define its **boundary**: $\partial m^\star := \mathrm{in}(m^\star) \cup \mathrm{out}(m^\star)$. The boundary condition $\partial m^\star \subseteq V^{\mathrm{obs}}$ is the structural property under which T4's formula transfers.

### 7.2 Theorem T6

**Theorem T6.** Let $\mathcal{M}$ be a HADMG with hidden variables. If $\partial m^\star \subseteq V^{\mathrm{obs}}$ then $\mathrm{do}(\neg m^\star)$ is identifiable from $P(V^{\mathrm{obs}})$ via T2's formula.

**Proof.** Two facts compose: (T6a) under noise independence, the conditional $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ in the full joint equals the causal mechanism factor (this is Fact T6a, requiring only C2); (T6b) under the boundary condition, the marginal conditional in $P(V^{\mathrm{obs}})$ equals the full-joint conditional (Fubini). Apply T2 in its full-joint form, marginalize, and use Fact T6b to pull the denominator outside the integral. Full proof: `THEOREM_H1_PLUS.md` §2.

T6 is the cleanest form of the H1+ result: T4's formula extends *verbatim* to hidden-variable HADMGs, as long as the target mechanism's boundary is observed. The hidden structure is preserved in the surviving factors of $P(V^{\mathrm{obs}})$ and propagates correctly to the post-intervention observed marginal.

### 7.3 Theorem T7: the boundary-violating reduction

**Theorem T7.** When $\partial m^\star \not\subseteq V^{\mathrm{obs}}$, identifiability of $\mathrm{do}(\neg m^\star)$ from $P(V^{\mathrm{obs}})$ reduces to identifying a stochastic-intervention marginal in the bipartite-blowup ADMG $B^\dagger(\mathcal{M})$, governed by Shpitser-Pearl 2006's hedge criterion.

**Proof sketch.** Mechanism deletion in $\mathcal{M}$ corresponds, under the bipartite blowup, to a stochastic intervention on $\mathrm{out}(m^\star)$ in $B^\dagger$ — assigning each output its $P_0$ distribution. Stochastic interventions reduce to standard $\mathrm{do}$ for identifiability purposes (Bareinboim-Pearl 2016). The post-intervention observed marginal is identifiable in $\mathcal{M}$ iff the corresponding marginal is identifiable in $B^\dagger$. Full sketch: `THEOREM_H1_PLUS.md` §3.

T7 is again a *reduction*: the boundary-violating case contains nothing new in principle — Pearl's existing machinery, applied to the right Pearl ADMG, settles the question.

### 7.4 The hyper-hedge

The structural obstruction to which T7 reduces is the **hyper-hedge**: a pair of sub-hypergraphs $(\mathcal{F}, \mathcal{F}')$ of $B^\dagger(\mathcal{M})$ analogous to Pearl's hedge but defined over hyper-c-components — maximal subsets of $V^{\mathrm{obs}}$ connected by latent-mechanism hyperedges (after Pearl-style latent projection of any hidden variables). Full definition: `THEOREM_H1_PLUS.md` §4.

When all mechanisms have $|\mathrm{out}(m)| = 1$, hyper-c-components reduce to Pearl c-components and hyper-hedges reduce to Pearl hedges (Shpitser-Pearl 2006, Definition 5).

### 7.5 Conjecture H1+ (full)

**Conjecture (Hyper-hedge completeness).** $\mathrm{do}(\neg m^\star)$ is identifiable from $P(V^{\mathrm{obs}})$ in $\mathcal{M}$ if and only if there is no hyper-hedge for $m^\star$ in $B^\dagger(\mathcal{M})$.

**Status.** The forward direction follows from T6 + T7 + Shpitser-Pearl soundness. The reverse direction (hyper-hedge ⇒ unidentifiable) lifts Shpitser-Pearl's completeness via the bipartite blowup; correctness depends on a deterministic-relations preservation lemma we have not proven. v1 commits to T6 (proved), T7 (sketched), and the hyper-hedge definition. Completeness is the principal remaining open problem; we expect it to require a focused effort comparable in technical depth to Shpitser-Pearl 2006 itself.

This is honest. The framework's foundations are settled; the deepest identifiability question is open.

---

## 8. Worked example: a 2-mechanism reaction network

We illustrate every preceding result on a single canonical example, deliberately chosen as the smallest system that exhibits genuine higher-order causal structure. Two coupled stoichiometric reactions:

- $m_1: A + B \to C + D$ with rate constant $k_1 = 0.5$
- $m_2: C + E \to F$ with rate constant $k_2 = 0.3$

Variables $V = \{A, B, C, D, E, F\}$; mechanisms $E = \{m_1, m_2\}$; exogenous $\{A, B, E\} \sim \mathrm{Unif}(1, 5)$. The structural functions:

$$
f_{m_1}(A, B, u_1) = (C, D) \text{ where } C = D = k_1 \cdot AB + u_1, \quad u_1 \sim \mathcal{N}(0, \sigma_1^2)
$$

$$
f_{m_2}(C, E, u_2) = F = k_2 \cdot CE + u_2, \quad u_2 \sim \mathcal{N}(0, \sigma_2^2)
$$

The crucial design choice: $m_1$ produces both $C$ and $D$ from a *single* noise $u_1$. The structural identity $C \equiv D$ is enforced by the mechanism, not as an emergent property of independent edges. This is the irreducibility witness: any Pearl SCM with independent equations $C := f_C(\cdot, u_C)$, $D := f_D(\cdot, u_D)$ and $u_C \perp u_D$ produces $\mathrm{Pr}(C = D) < 1$ generically. Recovering the structural identity in Pearl requires introducing a latent — and that latent *is* $m_1$ in our framework, but named, intervenable, and structurally typed.

### 8.1 Numerical verifications

Every claim below is reproduced by `minimal_model/test_*.py`:

**Observational distribution.** $\mathbb{E}[C] = \mathbb{E}[D] = 4.5$; $\mathrm{Corr}(C, D) = 1$ exactly (irreducibility witness, T2 §3).

**Three interventions.** $\mathrm{do}(A = 4)$ preserves the joint identity. $\mathrm{do}(\neg m_1)$ orphans $C, D$ to $\delta_0$; $F$ becomes pure noise $\mathcal{N}(0, \sigma_2^2)$. $\mathrm{do}(m_1 \to m_1')$ with $k_1' = 1.0$ doubles $\mathbb{E}[C], \mathbb{E}[D]$ while preserving $C = D$.

**Counterfactual.** Observed $(A=2, B=3, E=4, C=D=3.05, F=3.71)$. Abducted $u_1^* = 0.05$, $u_2^* = 0.05$. Under $\mathrm{do}(A = 4)$: $C^{\mathrm{cf}} = D^{\mathrm{cf}} = 6.05$, $F^{\mathrm{cf}} = 7.31$. The shared $u_1^*$ propagates identically to both counterfactual outputs, preserving $C^{\mathrm{cf}} = D^{\mathrm{cf}}$ — a property no Pearl SCM with independent noise can express.

### 8.2 Latent-mechanism extension (T4)

Add $m_{\mathrm{lat}}: \emptyset \to B, E$, a latent mechanism with bivariate-Gaussian noise inducing correlation $\rho = 0.6$ between $B$ and $E$. T4 says $\mathrm{do}(\neg m_1)$ remains identifiable. Direct simulation confirms: the latent confounding survives the intervention (post-intervention $\mathrm{Corr}(B, E) \approx 0.6$ is preserved in the surviving $P(B, E)$ factor), $C$ and $D$ orphan to $\delta_0$, $F$ becomes $\mathcal{N}(0, \sigma_2^2)$.

The semantically richest query is $\mathrm{do}(\neg m_{\mathrm{lat}})$ — intervening on the *latent* mechanism. T4 says this is also identifiable: $P^{\neg m_{\mathrm{lat}}}(V) = P(V) / P(B, E) \cdot \delta_0(B) \delta_0(E)$. The mechanism factor $P(B, E)$ is observable as the joint marginal even though $f_{m_{\mathrm{lat}}}$ is unknown. This query has no direct Pearl analogue (Pearl latents are unaddressable as intervention targets).

### 8.3 Hidden-variable extension (T6)

Add a hidden variable $W$ produced by a latent mechanism $m_W: \emptyset \to W$, and let $W$ be an input to a modified $m_2: C, E, W \to F$. T6 verdicts:

- $m_1$: $\partial m_1 = \{A, B, C, D\} \subseteq V^{\mathrm{obs}}$ → identifiable via T6.
- $m_2$: $\partial m_2 = \{C, E, W, F\}$, $W \in V^{\mathrm{lat}}$ → T6 conservatively rejects.
- $m_W$: $\partial m_W = \{W\} \subseteq V^{\mathrm{lat}}$ → T6 rejects.

Direct simulation shows that $\mathrm{do}(\neg m_2)$ and $\mathrm{do}(\neg m_W)$ are nonetheless identifiable in fact (T7's reduction succeeds). The gap between v1's T6-based verdict and the true identifiability status is exactly the open content of the hyper-hedge completeness conjecture.

---

## 9. Open problems and future directions

The v1 contributions (T1, Lemma 1.1, T2, T3, T4, T5, T6, T7) form a closed core under v1 conventions. We catalog the natural extensions in roughly increasing technical depth.

### 9.1 Hyper-hedge completeness (H1+)

The principal open problem is the reverse direction of T7: every hyper-hedge in $B^\dagger(\mathcal{M})$ obstructs identifiability. The forward direction is settled. The reverse requires lifting Shpitser-Pearl 2006's completeness proof, with a delicate deterministic-relations preservation argument under the bipartite blowup. This is a focused future effort.

### 9.2 The H2 conjecture, reassessed

We originally framed a conjecture H2: mechanism-correlated noise generates conditional independencies inexpressible by any finite-latent Pearl SCM. On reflection during v1, we judge this likely false in both distributional and interventional formulations — Pearl with sufficient latents matches any joint distribution by standard Bayesian-network universality, and stochastic-intervention reductions cover the interventional case. A salvageable reframing as a complexity-theoretic separation (smallest matching Pearl SCM is exponential in $|V|$) is conceivable but lacks a candidate construction. We retain H2 as a noted unresolved question rather than an active research direction.

### 9.3 Cyclic mechanism graphs

Allowing $G_E$ to contain cycles (reversible reactions, feedback regulation) requires fixed-point or measure-theoretic semantics. Bongers-Forré-Mooij 2018 provides Pearl-side machinery; lifting to hypergraphs is straightforward in form, demanding in execution.

### 9.4 Markov-kernel mechanisms

Replace each $f_m$ with a Markov kernel $K_m : \mathrm{Dom}(\mathrm{in}(m)) \to \Delta(\mathrm{Dom}(\mathrm{out}(m)))$. Subsumes deterministic-with-noise but breaks the noise-recovery formulation of counterfactual abduction. The natural setting is Markov categories (Fritz 2020).

### 9.5 Richer role typing

Generalize $\rho : E \to 2^V \times 2^V$ to assign typed roles (substrate / enzyme / product / cofactor) with appropriate equivariance constraints. Modest formal cost; substantial domain-fidelity gain for biological applications.

### 9.6 Causal abstraction as quotient

The Beckers-Halpern 2019 abstraction conditions naturally describe collapsing sub-hypergraphs into single mechanism nodes. Causal abstraction is an internal quotient operation in our framework; in Pearl it sits outside. Formalizing this should be straightforward and would clarify several v1 design choices.

### 9.7 Causal hypergraph transformers

A direction connecting the framework to hypergraph machine learning. AllSet (Chien 2022) and related architectures attend over typed incidence; a *causal* hypergraph transformer would respect the input/output role structure of mechanisms and encode mechanism-deletion / replacement as structured masking operations. This is the bridge between the formal causal theory of this paper and the descriptive hypergraph-learning literature reviewed in §2.4.

---

## 10. Conclusion

We have developed a strict generalization of Pearl's structural causal models in which mechanisms — typed hyperedges — are first-class causal objects. The framework retains all of Pearl's expressive power and adds two new intervention operations: mechanism deletion and mechanism replacement. The principal formal claim is the asymmetry of Theorem T4 and its hidden-variable extension T6: mechanism-level interventions admit a *closed-form identifier* read directly from the chain-rule factorization, while variable interventions reduce to Pearl multi-variable ID on the bipartite blowup. Under v1 conventions both interventions are likely identifiable in concrete cases; the asymmetry is closed-form-vs-case-analytic, not identifiable-vs-not. Under hidden variables it sharpens — T6 closes the observed-boundary case cleanly, while only T7's boundary-violating case (governed by the hyper-hedge) can genuinely fail.

The framework's value proposition is therefore not new expressive power but **first-class addressability**. Naming mechanisms as primary causal objects matches the structure of natural experiments (drug ablation, pathway knockout, n-ary policy intervention), simplifies the identifiability theory for those experiments, and unifies operations that Pearl can express only as multi-variable interventions or via latent-variable encoding.

The principal remaining open problem is hyper-hedge completeness — the reverse direction of T7's reduction. Several natural extensions (cyclic graphs, Markov kernels, richer roles, causal abstraction, causal hypergraph transformers) are sketched in §9. The reference implementation, totaling 60 verified test cases across five suites, supports every numerical claim in this paper and is provided as supplementary material.

The framework is small but, we believe, foundational. Its core technical content fits in a single paper; its extensions could occupy several more.

---

## References

Bareinboim, E., & Pearl, J. (2016). Causal inference and the data-fusion problem. *Proceedings of the National Academy of Sciences*, 113(27), 7345–7352.

Beckers, S., & Halpern, J. Y. (2019). Abstracting causal models. *AAAI 33*.

Bongers, S., Forré, P., Peters, J., & Mooij, J. M. (2018). Foundations of structural causal models with cycles and latent variables. *arXiv:1611.06221*.

Chien, E., Pan, C., Peng, J., & Milenkovic, O. (2022). You are AllSet: A multiset function framework for hypergraph neural networks. *ICLR 2022*.

Feinberg, M. (2019). *Foundations of Chemical Reaction Network Theory*. Springer.

Fritz, T. (2020). A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics. *Advances in Mathematics*, 370, 107239.

Gao, Y., Feng, Y., Ji, S., & Ji, R. (2022). HGNN+: General hypergraph neural networks. *IEEE TPAMI*.

Geiger, D., & Pearl, J. (1990). On the logic of causal models. *Uncertainty in Artificial Intelligence 4*.

Geiger, D., Verma, T., & Pearl, J. (1990). Identifying independence in Bayesian networks. *Networks*, 20(5), 507–533.

Halpern, J. Y. (2016). *Actual Causality*. MIT Press.

Jacobs, B., Kissinger, A., & Zanasi, F. (2019). Causal inference by string diagram surgery. *FoSSaCS 22*.

Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.

Kim, J., et al. (2022). Pure transformers are powerful graph learners. *NeurIPS 35*.

Murata, T. (1989). Petri nets: Properties, analysis and applications. *Proceedings of the IEEE*, 77(4), 541–580.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Rubenstein, P. K., et al. (2017). Causal consistency of structural equation models. *UAI 33*.

Shpitser, I., & Pearl, J. (2006). Identification of conditional interventional distributions. *UAI 22*.

Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.

Tian, J., & Pearl, J. (2002). A general identification condition for causal effects. *AAAI 18*.

Verma, T., & Pearl, J. (1990). Causal networks: Semantics and expressiveness. *Uncertainty in Artificial Intelligence 4*.

---

## Appendix A. Reference implementation

A Python implementation of v1 conventions is provided in `minimal_model/`. Approximately 800 lines, NumPy-only, no other dependencies. Public API: `HypergraphSCM`, `Mechanism`, `Factor`, `reaction_network()`, `reaction_network_with_latent_BE()`, `reaction_network_with_hidden_W()`, `d_separated()`, `deterministic_closure()`. Five test suites totaling 60 tests verify every numerical claim in this paper.

Run: `python -m minimal_model.test_example` (and analogously for `test_dseparation`, `test_factorization`, `test_hadmg`, `test_h1_plus`).

## Appendix B. Notation summary

| Symbol | Meaning |
|---|---|
| $V$ | variables (nodes) |
| $E$ | mechanisms (hyperedges) |
| $\rho$ | typed incidence assignment |
| $\mathrm{in}(m), \mathrm{out}(m)$ | input and output sets of $m$ |
| $\partial m$ | boundary of $m$, $\mathrm{in}(m) \cup \mathrm{out}(m)$ |
| $f_m$ | joint structural function of $m$ |
| $u_m$ | per-mechanism noise |
| $V^{\mathrm{exo}}$ | variables produced by no mechanism |
| $V^{\mathrm{obs}}, V^{\mathrm{lat}}$ | observed and latent variables (HADMG) |
| $E^{\mathrm{obs}}, E^{\mathrm{lat}}$ | observed and latent mechanisms (HADMG) |
| $P_0(v)$ | fallback exogenous distribution for $v$ |
| $G_E$ | mechanism dependency graph |
| $B(\mathcal{M})$ | bipartite blowup |
| $B^\dagger(\mathcal{M})$ | bipartite-blowup ADMG |
| $\mathcal{I}(\mathcal{M})$ | intervention space |
| $\mathrm{Det}_{\mathcal{M}}(Z)$ | functional-determination closure of $Z$ |
| $Z^* = Z \cup \mathrm{Det}(Z)$ | augmented conditioning set |
| $M$ | typed incidence matrix |

## Appendix C. Theorem dependency graph

```
                  Lemma 1.1 (mechanism-level chain rule)
                          |
                  +-------+-------+
                  |               |
              Theorem T2     Theorem T3
              (deletion)    (replacement)
                  |               |
                  +-------+-------+
                          |
                  Theorem T4 (HADMG; observed variables)
                          |
                  +-------+-------+
                  |               |
              Theorem T5     Theorem T6
              (variable     (hidden variables;
              reduction)    observed boundary)
                                  |
                          Theorem T7 (boundary violation)
                                  |
                          [hyper-hedge def]
                                  |
                          [H1+ completeness — open]


              Theorem T1 (d*-separation)  — independent foundation,
                                            used implicitly throughout
```

All theorems through T7 are proved or proof-sketched in the supplementary documents (`THEOREM_T1.md`, `THEOREM_T2_T3.md`, `THEOREM_T4_T5.md`, `THEOREM_H1_PLUS.md`). Numerical claims are verified by the test suites.
