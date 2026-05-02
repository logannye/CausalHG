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

## 5. The principal open problem: identifiability under hidden mechanisms

When *not* causally sufficient — when there exist unobserved mechanisms $m^{\mathrm{lat}}$ in the true generative model — Lemma 1.1 fails to give clean factors. Two distinct failure modes arise, only one of which has a Pearl analogue:

### 5.1 Hidden-mechanism confounding (Pearl-analogous)

A latent mechanism $m^{\mathrm{lat}}$ with outputs in $V$ creates the standard form of confounding: $P(\mathrm{out}(m) \mid \mathrm{in}(m))$ no longer equals the observational conditional. Pearl's ID algorithm (Tian-Pearl 2002, Shpitser-Pearl 2006) generalizes to this setting via a hypergraph analogue of the *hedge* obstruction. **Conjecture H1.** A hypergraph ID algorithm with hedge-obstruction completeness exists and is computable in polynomial time on $|V| + |E|$.

### 5.2 Mechanism-correlated noise (new — no Pearl analogue)

Even with no latent mechanism, the noise terms $\{u_m\}$ may be correlated in reality (e.g., shared environmental fluctuation, common stochastic input). The independence assumption $u_{m_1} \perp u_{m_2}$ is part of v1 (convention C2 spirit) but is empirically violatable.

This form of confounding is **strictly hypergraphic**: in a Pearl SCM with single-output equations, noise correlation between distinct equations is mathematically equivalent to a latent common cause (which can always be made explicit). In a hypergraph SCM with joint mechanisms, noise correlation between *different mechanisms* cannot be reduced to a latent variable — it is a constraint at the *mechanism level*, not the variable level.

**Conjecture H2.** Mechanism-correlated noise generates conditional independencies in $P^{\mathcal{M}}$ that cannot be captured by *any* enlarged Pearl SCM with finitely many additional latent variables. The hypergraph framework is the minimal formalism in which this constraint is expressible.

If H2 holds, the hypergraph framework's contribution is not just notational convenience but **strictly greater representational capacity at the level of identifiable structure**.

### 5.3 Identifiability with mechanism-correlated noise

Under hidden-mechanism *and* mechanism-correlated-noise confounding, identifiability becomes a richer problem with — to our knowledge — no analog in the Pearl literature. The conjectured resolution is a two-tier identifiability theory:

- Tier 1 (standard): a hedge-obstruction criterion analogous to Shpitser-Pearl, addressing latent mechanisms.
- Tier 2 (new): a *correlation criterion* on mechanism-noise covariances, addressing mechanism-correlated noise.

Developing these is the principal theoretical work remaining in the project. The v1 implementation handles only the causally-sufficient case (Tier 0).

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

## 7. Why mechanism-level interventions are *more* identifiable than Pearl multi-variable

A subtle but important point. Pearl's $\mathrm{do}(\neg m_1)$-equivalent intervention is the *simultaneous* multi-variable intervention $\mathrm{do}(C = c, D = c)$ for some $c$, integrated against the joint of $(C, D)$ that $m_1$ would have produced. In Pearl, multi-variable interventions are identifiable under more restrictive conditions than single-variable ones (Tian-Pearl 2002 §4). The hypergraph framework, by contrast, treats $\mathrm{do}(\neg m_1)$ as a *single* intervention and gives it identifiability under the *same* conditions as variable interventions.

This is a substantive theoretical observation: **first-class addressability of mechanisms simplifies identifiability theory.** The identifiability of mechanism-level interventions in the hypergraph framework is uniformly at least as good as, and sometimes strictly better than, the identifiability of their Pearl multi-variable translations.

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
            (Open) H1: hedge-obstruction ID algorithm
            (Open) H2: mechanism-correlated noise expressibility
```

---

## References

- Pearl, J. (1995). "Causal diagrams for empirical research." *Biometrika* 82.
- Tian, J. & Pearl, J. (2002). "A general identification condition for causal effects." *AAAI*.
- Shpitser, I. & Pearl, J. (2006). "Identification of conditional interventional distributions." *UAI*.
- Bareinboim, E. & Pearl, J. (2016). "Causal inference and the data-fusion problem." *PNAS* 113.
