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

**Theorem T2 (kernel form).** Under v1 conventions and causal sufficiency, for any $m^\star \in E$ with fallback policy $P_0^{m^\star}$ on the orphaned outputs $\mathrm{out}(m^\star)$:

$$
P^{\mathcal{M}^{\neg m^\star}}(V) \;=\; \left[\prod_{v \in V^{\mathrm{exo}}} P(v)\right] \cdot \left[\prod_{m \in E \setminus \{m^\star\}} P\!\left(\mathrm{out}(m) \mid \mathrm{in}(m)\right)\right] \cdot P_0^{m^\star}\!\left(\mathrm{out}(m^\star)\right).
$$

*Proof.* By Lemma 1.1, $P^{\mathcal{M}}(V)$ has factor $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ for $m^\star$. Deleting $m^\star$ removes this factor from the chain-rule product. Under C4, every $v \in \mathrm{out}(m^\star)$ has no other producer, hence becomes orphaned; the whole set $\mathrm{out}(m^\star)$ acquires the single joint factor $P_0^{m^\star}$. The remaining factors are unchanged, so Lemma 1.1 applied to $\mathcal{M}^{\neg m^\star}$ gives the display above. $\square$

**Corollary T2.1 (quotient form).** If additionally $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)) > 0$ holds $P^{\mathcal{M}^{\neg m^\star}}$-almost everywhere, then

$$
P^{\mathcal{M}^{\neg m^\star}}(V) \;=\; \frac{P^{\mathcal{M}}(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot P_0^{m^\star}\!\left(\mathrm{out}(m^\star)\right).
$$

*Proof.* Multiply and divide the kernel form by $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ — legitimate exactly on the region where that factor is nonzero — and apply Lemma 1.1 in reverse. $\square$

**Remark T2.2 (the positivity hypothesis is not cosmetic).** The quotient form is the one that reads like a truncated factorization, but it is the weaker statement, and its hypothesis fails in precisely the case this framework exists to model.

C2 posits deterministic structural functions driven by exogenous noise. When $u_{m^\star}$ carries fewer degrees of freedom than $|\mathrm{out}(m^\star)|$ — which is what stoichiometric coupling *is* — the mechanism factor is singular, supported on a proper subset of $\mathrm{Dom}(\mathrm{out}(m^\star))$. Deleting $m^\star$ replaces that factor by one with full support, so $P^{\mathcal{M}^{\neg m^\star}}$ puts mass exactly where $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star))$ vanishes. On that region the quotient is $0/0$, while the kernel form is defined and correct.

This is not a measure-theoretic technicality about continuous densities: it is already visible in a two-state discrete model. Take $\mathrm{out}(m^\star) = \{C, D\}$ binary with $C \equiv D$, and $P_0^{m^\star}$ of full support. Then $P(C \neq D \mid A,B) = 0$ while $P^{\neg m^\star}(C \neq D) > 0$.

§5 below carries the continuous instance of the same fact, and states it outright: the worked example's mechanism factor is "singular with respect to Lebesgue measure on $\mathbb{R}^2$". The example nonetheless reaches the right answer under the quotient form only because it takes $P_0^{m_1} = \delta_0 \otimes \delta_0$, which is supported on the diagonal $C = D$ where the singular factor lives. A fallback with any off-diagonal mass breaks it.

Accordingly, the compiler emits the kernel form whenever every chain-rule factor is observational, and falls back to the quotient only where hidden variables leave no alternative (T6), recording `Target positivity` explicitly when it does.

---

## 3. Theorem T3: mechanism replacement

**Theorem T3 (kernel form).** Under v1 conventions and causal sufficiency, for any $m^\star \in E$ and any replacement $m'$ with $\rho(m') = \rho(m^\star)$:

$$
P^{\mathcal{M}^{m^\star \to m'}}(V) \;=\; \left[\prod_{v \in V^{\mathrm{exo}}} P(v)\right] \cdot \left[\prod_{m \in E \setminus \{m^\star\}} P\!\left(\mathrm{out}(m) \mid \mathrm{in}(m)\right)\right] \cdot P_{f_{m'}}\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)
$$

where $P_{f_{m'}}$ is the conditional distribution induced by the replacement structural function $f_{m'}$ and noise $u_{m'}$.

*Proof.* Replacement substitutes the mechanism factor for $m^\star$ with the corresponding factor for $m'$, leaving all other factors untouched (no orphaning, since the new mechanism produces the same variables). Apply Lemma 1.1 to $\mathcal{M}^{m^\star \to m'}$. $\square$

**Corollary T3.1 (quotient form).** If $P(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)) > 0$ holds $P^{\mathcal{M}^{m^\star \to m'}}$-almost everywhere, the same multiply-and-divide step yields

$$
P^{\mathcal{M}^{m^\star \to m'}}(V) \;=\; \frac{P^{\mathcal{M}}(V)}{P\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right)} \cdot P_{f_{m'}}\!\left(\mathrm{out}(m^\star) \mid \mathrm{in}(m^\star)\right).
$$

Remark T2.2 applies verbatim, and bites harder here: $\rho(m') = \rho(m^\star)$ constrains incidence, not behaviour, so replacing a coupled mechanism with a decoupled one is a legal query — "what if this complex were two independent enzymes?" — and it is exactly a query whose new mass lands where the old factor vanishes.

**Corollary T3.2 (deletion as replacement).** $\mathrm{do}(\neg m^\star)$ is the special case of $\mathrm{do}(m^\star \to m')$ where $m'$ is the *trivial mechanism* whose conditional distribution equals $P_0^{m^\star}(\mathrm{out}(m^\star))$, independent of inputs.

This unifies the two new operations: deletion is the "set this mechanism to its $P_0$-default" form of replacement.

**Remark T3.3 (why $P_0$ is joint, and what the product form silently asserted).** Corollary T3.2 makes the type of $P_0$ visible. An earlier version of this document defined deletion as replacement by a *product* of per-variable fallbacks, $\prod_{v \in \mathrm{out}(m^\star)} P_0(v)$. That is not a neutral choice of notation: it makes $\mathrm{do}(\neg m^\star)$ render $\mathrm{out}(m^\star)$ mutually independent as a matter of definition. A framework whose stated motivation is that jointly produced outputs are structurally coupled could then not express a post-deletion law preserving any coupling — for instance a knocked-out reaction whose products still satisfy a conservation constraint imposed by something other than $m^\star$. The restriction was invisible in the worked example of §5 precisely because that example's fallback, $\delta_0 \otimes \delta_0$, happens to factorize.

Nothing in Lemma 1.1 or T2 requires it. The proofs use only that the inserted factor is a fixed kernel over $\mathrm{out}(m^\star)$ not depending on the rest of the model; whether it factorizes is never invoked. Accordingly $P_0^{m^\star}(\mathrm{out}(m^\star))$ is a single joint kernel throughout this document, and $\prod_v P_0(v)$ is recovered as its independent special case, so no previously expressible model is lost.

Two type distinctions are worth stating explicitly, since both were blurred by the product form:

- $P_0^{m^\star}$ carries **no conditioning on $\mathrm{in}(m^\star)$**. Deletion removes the mechanism, so its inputs no longer act. A policy that still reads the inputs is $\mathrm{do}(m^\star \to m')$, not $\mathrm{do}(\neg m^\star)$, and belongs to T3.
- $P_0^{m^\star}$ is indexed by the **mechanism**, not by its output set. Under C4 the two are in bijection, so this costs nothing; it is what keeps the policy attached to the intervention that installs it.

This is a strictly weaker hypothesis than the one it replaces, so every theorem in this document, in `THEOREM_T4_T5.md`, and in `THEOREM_H1_PLUS.md` holds verbatim.

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

Apply T2 with $P_0^{m_1} = \delta_0 \otimes \delta_0$:

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

A subtle but important point. Pearl's $\mathrm{do}(\neg m_1)$-equivalent intervention is the *stochastic multi-variable* intervention $\mathrm{do}(C \sim P_0, D \sim P_0)$, with $(C, D)$ jointly resampled from $P_0^{m_1}$. In Pearl ADMGs, identifying multi-variable stochastic interventions reduces to standard multi-variable ID with a substitution step (Bareinboim-Pearl 2016) and is in general case-analytic — the ID algorithm runs, may invoke the do-calculus rules in non-trivial sequences, and (in the worst case) returns a hedge.

The hypergraph framework, by contrast, treats $\mathrm{do}(\neg m_1)$ as a *single* operation and gives it a **closed-form identifier** read directly from Lemma 1.1's factorization: $P(V) / P(\mathrm{out}(m_1) \mid \mathrm{in}(m_1)) \cdot P_0^{m_1}(\mathrm{out}(m_1))$. No algorithmic search, no case analysis, no hedge check — the formula is uniform in the structure of $\mathcal{M}$.

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
