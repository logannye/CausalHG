# Theorems T2 and T3: Mechanism-Level Identifiability

This document states and proves identifiability results for the two new operations introduced by the hypergraph framework: mechanism deletion $\mathrm{do}(\neg m)$ and mechanism replacement $\mathrm{do}(m \to m')$.

The results are clean under **causal sufficiency** (no unobserved mechanisms). Identifiability under hidden mechanisms — including a new form of confounding unique to the hypergraph framework — is flagged as the principal open problem (§5).

---

## 0. Setup and the additional convention

We work under v1 conventions C1–C3 (`FOUNDATIONS.md` §0). T2 requires one further convention:

- **C4. Single producer.** For all distinct $m_1, m_2 \in E$: $\mathrm{out}(m_1) \cap \mathrm{out}(m_2) = \emptyset$. Each variable has at most one producing mechanism.

C4 is the hypergraph analogue of Pearl's "one structural equation per variable." It rules out variables jointly produced by multiple mechanisms — a configuration that is conceptually problematic (it is unclear how competing mechanisms compose) and rare in practice. C4 is enforced by `HypergraphSCM.validate()` in the reference implementation.

We also require **causal sufficiency** for v1:

- **Causal sufficiency.** Every mechanism that affects $V$ is in $E$ (no unobserved mechanisms).

Under causal sufficiency and v1 conventions, the per-mechanism noise $u_m$ is independent of all other variables and noise — the standard Markov assumption, lifted to mechanisms.

---

## 1. The chain-rule factorization

**Lemma 1.1 (Mechanism-level chain rule).** Under v1 conventions C1–C4 and causal sufficiency,

$$
P^{\mathcal{M}}(V) \;=\; \left[\prod_{v \in V^{\mathrm{exo}}} P(v)\right] \cdot \left[\prod_{m \in E} P\!\left(\mathrm{out}(m) \;\middle|\; \mathrm{in}(m)\right)\right].
$$

*Proof.* Sample $\mathcal{M}$ in topological order of $G_E$ (C1). Each $v \in V^{\mathrm{exo}}$ is drawn independently with marginal $P(v)$. Each mechanism $m$, given its inputs (already realized by topological order), produces $\mathrm{out}(m)$ as a deterministic function of $\mathrm{in}(m)$ and independent noise $u_m$; the conditional distribution of the joint output is therefore $P(\mathrm{out}(m) \mid \mathrm{in}(m))$ — the pushforward of $P(u_m)$ through $f_m(\mathrm{in}(m), \cdot)$. C4 ensures these factors do not overlap (no variable appears as output of multiple factors). Independence of noise across mechanisms makes the factors conditionally independent given the topological-prefix evidence. □

The mechanism-level chain rule is the principal new factorization tool. It generalizes Pearl's variable-level factorization $P(V) = \prod_v P(v \mid \mathrm{pa}(v))$ to *joint conditional distributions over mechanism outputs*. Each factor $P(\mathrm{out}(m) \mid \mathrm{in}(m))$ is the **mechanism factor** of $m$.

---

## 2. Theorem T2: mechanism-deletion truncation

**Theorem T2.** Under v1 conventions and causal sufficiency, for any $m^\star \in E$ with fallback $P_0$ on the orphaned outputs $\mathrm{out}(m^\star)$:

$$
P^{\mathcal{M}^{\neg m^\star}}(V) \;=\; \frac{P^{\mathcal{M}}(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v).
$$

*Proof.* By Lemma 1.1, $P^{\mathcal{M}}(V)$ has factor $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ for $m^\star$. Deleting $m^\star$ removes this factor from the chain-rule product. Under C4, every $v \in \mathrm{out}(m^\star)$ has no other producer, hence becomes orphaned and acquires the $P_0$ factor. The remaining factors are unchanged. The post-intervention chain rule gives

$$
P^{\mathcal{M}^{\neg m^\star}}(V) = \left[\prod_{v \in V^{\mathrm{exo}}_{\mathrm{orig}}} P(v)\right] \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v) \cdot \prod_{m \in E \setminus \{m^\star\}} P(\mathrm{out}(m) \mid \mathrm{in}(m)).
$$

Multiplying numerator and denominator by $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ and applying Lemma 1.1 in reverse:

$$
P^{\mathcal{M}^{\neg m^\star}}(V) = \frac{P^{\mathcal{M}}(V)}{P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))} \cdot \prod_{v \in \mathrm{out}(m^\star)} P_0(v). \qquad \square
$$

---

## 3. Theorem T3: mechanism replacement

**Theorem T3.** Under v1 conventions and causal sufficiency, for any $m^\star \in E$ and any replacement $m'$ with $\rho(m') = \rho(m^\star)$:

$$
P^{\mathcal{M}^{m^\star \to m'}}(V) \;=\; \frac{P^{\mathcal{M}}(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot P_{f_{m'}}\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)
$$

where $P_{f_{m'}}$ is the conditional distribution induced by the replacement structural function $f_{m'}$ and noise $u_{m'}$.

*Proof.* Replacement substitutes the mechanism factor for $m^\star$ with the corresponding factor for $m'$, leaving all other factors untouched (no orphaning, since the new mechanism produces the same variables). Apply Lemma 1.1. □

**Corollary T3.1.** $\mathrm{do}(\neg m^\star)$ is the special case of $\mathrm{do}(m^\star \to m')$ where $m'$ is the *trivial mechanism* whose conditional distribution equals the product of fallbacks $\prod_{v \in \mathrm{out}(m^\star)} P_0(v)$, independent of inputs.

This corollary unifies the two new operations: deletion is the "set this mechanism to its $P_0$-default" form of replacement.

---

## 4. Identifiability under causal sufficiency

**Corollary T2.1 (Identifiability).** Under v1 conventions and causal sufficiency, $\mathrm{do}(\neg m^\star)$ and $\mathrm{do}(m^\star \to m')$ are identifiable from $P^{\mathcal{M}}$ — i.e., their post-intervention distributions can be computed from the observational distribution alone — provided:

(a) The mechanism factor $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is identifiable from $P^{\mathcal{M}}$, which under causal sufficiency reduces to the standard observational conditional distribution.

(b) The fallback $P_0$ (for deletion) or replacement distribution $P_{f_{m'}}$ (for replacement) is part of the model specification.

Under causal sufficiency, (a) is automatic: $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ is directly readable from $P^{\mathcal{M}}$ by conditioning. (b) is part of the intervention's definition, not the observational data.

Hence: **mechanism-level interventions are identifiable under causal sufficiency.** The interesting case is when this fails, addressed in §5.

---

## 5. Beyond causal sufficiency

Causal sufficiency is restrictive. Two natural relaxations are interesting:

### 5.1 Hidden mechanisms (developed in `THEOREM_T4_T5.md`)

When some mechanisms in the true generative model are unobserved — their typed incidence known but their structural functions unknown — Lemma 1.1 still applies (its proof uses only C1–C4 and noise independence; it does not require observability of $f_m$). The mechanism factor $P(\mathrm{out}(m) \mid \mathrm{in}(m))$ remains a conditional in the observational distribution and is therefore directly observable. This is the content of T4 (`THEOREM_T4_T5.md` §1): mechanism-deletion identifiability extends verbatim to HADMGs with $V^{\mathrm{lat}} = \emptyset$. Identifiability under additionally hidden *variables* is treated by T6/T7 in `THEOREM_H1_PLUS.md`, with the hyper-hedge completeness conjecture as the principal open problem.

### 5.2 Mechanism-correlated noise (retracted as a strict-dominance claim)

The independence assumption $u_{m_1} \perp u_{m_2}$ for $m_1 \neq m_2$ is part of v1 (in the spirit of C2) but is empirically violatable: shared environmental fluctuation or common stochastic input can correlate the noise terms of distinct mechanisms.

An earlier draft of this document conjectured (**H2**) that mechanism-correlated noise generates conditional independencies inexpressible by *any* finite-latent Pearl SCM, and that the hypergraph framework would therefore have strictly greater representational capacity than Pearl-with-latents. **We retract this conjecture.** Pearl with sufficient latents is universal at both the distributional and interventional level (any joint distribution and any interventional query realized by an HSCM with correlated mechanism noise is realized by some Pearl SCM with appropriately introduced latent common causes). A salvageable reframing as a *complexity-theoretic* separation — that the smallest matching Pearl SCM has size exponential in $|V|$ or $|E|$ — is conceivable but lacks a candidate construction; we do not pursue it.

What remains true is a modeling-ergonomic point rather than a representational one: when noise is naturally correlated *across mechanisms*, the hypergraph formalism keeps the correlation at the level where the experimentalist actually reasons about it (the mechanism), whereas a Pearl encoding must introduce an auxiliary latent and then strip it away in interpretation. T4's clean closed form requires noise independence; under correlated mechanism noise, identification becomes an instance of standard hidden-confounder analysis on a Pearl ADMG obtained from the bipartite blowup, with no special hypergraph machinery needed.

---

## 6. Worked instance: $\mathrm{do}(\neg m_1)$ in the reaction network

Apply T2 to the example of `MINIMAL_EXAMPLE.md` with $m^\star = m_1$:

The chain-rule factorization (Lemma 1.1):

$$
P(V) = P(A) P(B) P(E) \cdot P(C, D \mid A, B) \cdot P(F \mid C, E).
$$

$m_1$'s mechanism factor is $P(C, D \mid A, B)$ — the conditional joint distribution of the two outputs of $m_1$. Under our parametrization:

$$
P(C, D \mid A, B) = \mathcal{N}_{\mathrm{deg}}\!\left(\binom{C}{D}; \binom{k_1 A B}{k_1 A B}, \begin{pmatrix} \sigma_1^2 & \sigma_1^2 \\ \sigma_1^2 & \sigma_1^2 \end{pmatrix}\right)
$$

— a degenerate bivariate Gaussian supported on the line $C = D$, with variance $\sigma_1^2$ along that line. The degeneracy is the formal expression of joint structure: $P(C, D \mid A, B)$ does not factor as $P(C \mid A, B) \cdot P(D \mid A, B)$, and is in fact singular with respect to Lebesgue measure on $\mathbb{R}^2$.

Apply T2 with $P_0(C) = P_0(D) = \delta_0$:

$$
P^{\neg m_1}(V) = \frac{P(V)}{P(C, D \mid A, B)} \cdot \delta_0(C) \delta_0(D) = P(A) P(B) P(E) \cdot \delta_0(C) \delta_0(D) \cdot P(F \mid C, E).
$$

Substituting $C = 0$ in the surviving factor:

$$
P^{\neg m_1}(V) = P(A) P(B) P(E) \cdot \delta_0(C) \delta_0(D) \cdot P(F \mid C = 0, E)
$$

where $P(F \mid C = 0, E) = \mathcal{N}(F; 0, \sigma_2^2)$ since $f_{m_2}(0, E, u_2) = u_2$.

This matches direct simulation of $\mathcal{M}^{\neg m_1}$ exactly, confirming T2 on the worked example. The numerical verification is in `minimal_model/test_factorization.py`.

---

## 7. Why mechanism-level interventions admit a closed-form identifier

A subtle but important point. Pearl's $\mathrm{do}(\neg m_1)$-equivalent intervention is the *stochastic multi-variable* intervention $\mathrm{do}(C \sim P_0, D \sim P_0)$, with $(C, D)$ jointly resampled from $\prod P_0$. In Pearl ADMGs, identifying multi-variable stochastic interventions reduces to standard multi-variable ID with a substitution step (Bareinboim-Pearl 2016) and is in general case-analytic — the ID algorithm runs, may invoke the do-calculus rules in non-trivial sequences, and (in the worst case) returns a hedge.

The hypergraph framework, by contrast, treats $\mathrm{do}(\neg m_1)$ as a *single* operation and gives it a **closed-form identifier** read directly from Lemma 1.1's factorization: $P(V) / P(\mathrm{out}(m_1) \mid \mathrm{in}(m_1)) \cdot \prod_{v \in \mathrm{out}(m_1)} P_0(v)$. No algorithmic search, no case analysis, no hedge check — the formula is uniform in the structure of $\mathcal{M}$.

This is the framework's substantive theoretical contribution at the level of identifiability: **first-class addressability of mechanisms collapses a multi-variable case-analytic ID problem into a single closed-form expression.** Whether this collapse also extends the *class* of identifiable queries — a stronger claim — depends on the setting. Under v1 conventions with $V^{\mathrm{lat}} = \emptyset$, we believe both formalisms reach the same identifiability verdicts in concrete cases (see `THEOREM_T4_T5.md` §3 for the precise observation). Under hidden variables, T6's observed-boundary closed form does in concrete cases bypass hyper-hedge analysis that Pearl's ID would otherwise require.

---

## 8. Summary of theorem dependencies

```
              Lemma 1.1 (chain rule)
                      |
            +---------+---------+
            |                   |
        Theorem T2          Theorem T3
        (deletion)          (replacement)
            |                   |
            +---------+---------+
                      |
            Corollary T2.1 (identifiability under sufficiency)
                      |
                      v
            T4 (hidden mechanisms; observed variables)
                      |
                      v
            T6 (hidden variables; observed boundary)
                      |
                      v
            T7 + hyper-hedge (boundary-violating reduction)
                      |
                      v
            (Open) H1+ completeness — `THEOREM_H1_PLUS.md` §4.3
```

---

## References

- Pearl, J. (1995). "Causal diagrams for empirical research." *Biometrika* 82.
- Tian, J. & Pearl, J. (2002). "A general identification condition for causal effects." *AAAI*.
- Shpitser, I. & Pearl, J. (2006). "Identification of conditional interventional distributions." *UAI*.
- Bareinboim, E. & Pearl, J. (2016). "Causal inference and the data-fusion problem." *PNAS* 113.
