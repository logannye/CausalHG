# Minimal Example: A 2-Mechanism Reaction Network (v0.1)

This document instantiates the v1 framework defined in `FOUNDATIONS.md` on the smallest system that exhibits genuine higher-order causal structure — irreducible to a Pearl SCM with single-output equations.

All hand calculations in this document are verified by `minimal_model/test_example.py`.

---

## 1. The system

Two coupled stoichiometric reactions:

- $m_1: A + B \to C + D$ with rate constant $k_1 = 0.5$
- $m_2: C + E \to F$ with rate constant $k_2 = 0.3$

Variables $V = \{A, B, C, D, E, F\}$. Mechanisms $E = \{m_1, m_2\}$.

Typed incidence:

| | $m_1$ | $m_2$ |
|---|---|---|
| $A$ | $-1$ | $0$ |
| $B$ | $-1$ | $0$ |
| $C$ | $+1$ | $-1$ |
| $D$ | $+1$ | $0$ |
| $E$ | $0$ | $-1$ |
| $F$ | $0$ | $+1$ |

Note $C$ has both signs — it is an output of $m_1$ and an input of $m_2$. This induces the mechanism dependency edge $m_1 \to m_2$ in $G_E$, which is acyclic (C1 holds).

Exogenous: $V^{\mathrm{exo}} = \{A, B, E\}$, with $A, B, E \sim \mathrm{Unif}(1, 5)$ independent.

Fallback distribution: $P_0(v) = \delta_0$ for all $v$ (concentration zero when no producing mechanism is active).

---

## 2. Structural equations

$$
f_{m_1}(A, B, u_1) = (C, D) \quad \text{where} \quad C = D = k_1 \cdot A \cdot B + u_1, \quad u_1 \sim \mathcal{N}(0,\, 0.1^2)
$$

$$
f_{m_2}(C, E, u_2) = F \quad \text{where} \quad F = k_2 \cdot C \cdot E + u_2, \quad u_2 \sim \mathcal{N}(0,\, 0.1^2)
$$

**The crucial design choice.** $m_1$ produces *both* $C$ and $D$ from a *single* noise term $u_1$. The structural identity

$$
C - D \equiv 0 \quad \text{(holds for all } A, B, u_1\text{)}
$$

is enforced *as part of the mechanism*, not as an emergent property of independent edges. This is the irreducibility witness.

This is the **over-determined abduction case** of `FOUNDATIONS.md` §8.1: $|\mathrm{out}(m_1)| = 2 > 1 = \dim(u_1)$. On-manifold (i.e., $C = D$), abduction recovers $u_1 = C - k_1 \cdot A \cdot B$ uniquely.

---

## 3. Observational distribution

By Proposition 5.1 (sampling equivalence), we sample $A, B, E$ exogenously, draw $u_1, u_2$, and evaluate mechanisms in topological order.

**Marginal moments of the joint output:**

$$
\mathbb{E}[C] = \mathbb{E}[D] = k_1 \, \mathbb{E}[A] \, \mathbb{E}[B] = 0.5 \cdot 3 \cdot 3 = 4.5
$$

$$
\mathrm{Var}(C) = \mathrm{Var}(D) = k_1^2 \, \mathrm{Var}(AB) + \sigma_1^2
$$

$$
\mathrm{Cov}(C, D) = \mathrm{Var}(C), \qquad \mathrm{Corr}(C, D) = 1 \text{ exactly}
$$

The exact $\mathrm{Corr}(C, D) = 1$ is the empirical signature of joint mechanism. **This signature is structurally inexpressible in any Pearl SCM** with independent equations $C := f_C(A, B, u_C)$, $D := f_D(A, B, u_D)$, $u_C \perp u_D$, unless both equations are deterministic — see §7.

---

## 4. Three interventions

### 4.1 Intervention 1: $\mathrm{do}(A = 4)$ — variable intervention

Remove no mechanisms (no producer of $A$ in $E$); add $A$ to $V^{\mathrm{exo}}$ with point mass at $4$.

Result: $C = D = 0.5 \cdot 4 \cdot B + u_1 = 2B + u_1$. $F$ unchanged in functional form.

This is a standard Pearl variable intervention; the bipartite blowup makes it Pearl-equivalent.

### 4.2 Intervention 2: $\mathrm{do}(\neg m_1)$ — *the new operation*

Delete mechanism $m_1$. Variables $C$ and $D$ have no remaining producer; they enter $V^{\mathrm{exo}}$ with $P_0 = \delta_0$. Then:

$$
C = D = 0, \qquad F = k_2 \cdot 0 \cdot E + u_2 = u_2 \sim \mathcal{N}(0,\, 0.01)
$$

**No Pearl analogue as a single intervention.** The closest Pearl translation requires *simultaneously* intervening $\mathrm{do}(C=0)$ AND $\mathrm{do}(D=0)$ — two interventions, not one. The distinction is experimentally meaningful: a single drug ablating an enzyme is one experiment; the Pearl translation is a multi-target simultaneous intervention.

This is the formal content of "mechanisms are first-class causal objects."

### 4.3 Intervention 3: $\mathrm{do}(m_1 \to m_1')$ — mechanism replacement

Replace $f_{m_1}$ with $f_{m_1'}$: $C = D = k_1' \cdot A \cdot B + u_1$ where $k_1' = 1.0$ (an enzyme variant with double the catalytic efficiency). Same incidence $\rho(m_1') = \rho(m_1)$, same noise distribution, different functional form.

Crucially, the joint output structure is preserved: $C^{m_1'} = D^{m_1'}$ always. This is what distinguishes a mechanism replacement from independent-edge replacement — replacing two Pearl edges $A \to C$ and $A \to D$ with two new edges does not preserve joint coupling unless they are explicitly linked by a shared latent.

---

## 5. Counterfactual: a fully-worked computation

**Factual observation:** $A = 2,\, B = 3,\, E = 4,\, C = D = 3.05,\, F = 3.71$.

Let $\sigma_1 = \sigma_2 = 0.1$.

### Step 1 — Abduction

Exogenous noise (just the variable values, since $A, B, E$ are exogenous):

$$
u^{\mathrm{exo}}_A = 2, \quad u^{\mathrm{exo}}_B = 3, \quad u^{\mathrm{exo}}_E = 4
$$

Mechanism noise (invert $f_m$ given inputs and outputs):

$$
u_1^* = C - k_1 A B = 3.05 - 0.5 \cdot 2 \cdot 3 = 3.05 - 3.00 = 0.05
$$

$$
u_2^* = F - k_2 C E = 3.71 - 0.3 \cdot 3.05 \cdot 4 = 3.71 - 3.66 = 0.05
$$

For $u_1$ to be well-defined, we need the over-determined consistency $C = D$ to hold in the factual — it does (both observed at $3.05$).

### Step 2 — Action

Apply $\mathrm{do}(A := 4)$, producing $\mathcal{M}^*$ with $A$ in $V^{\mathrm{exo}}$ at point mass $4$.

### Step 3 — Prediction

Re-evaluate $\mathcal{M}^*$ with abducted noise:

$$
A^{cf} = 4 \text{ (intervention)}, \quad B^{cf} = 3, \quad E^{cf} = 4 \text{ (carried)}
$$

$$
C^{cf} = D^{cf} = k_1 A^{cf} B^{cf} + u_1^* = 0.5 \cdot 4 \cdot 3 + 0.05 = 6.05
$$

$$
F^{cf} = k_2 C^{cf} E^{cf} + u_2^* = 0.3 \cdot 6.05 \cdot 4 + 0.05 = 7.31
$$

The shared $u_1^*$ propagates *identically* to both $C^{cf}$ and $D^{cf}$, preserving $C^{cf} = D^{cf}$. A Pearl-decomposed model with separate noise terms $u_C, u_D$ would have $C^{cf} \neq D^{cf}$ in general — the conservation evaporates under counterfactual.

---

## 6. Counterfactual under mechanism deletion

Same factual, but now ask: "had $m_1$ been knocked out, what would $F$ have been?"

### Step 1 — Abduction (as before)

$u^{\mathrm{exo}}_A = 2, u^{\mathrm{exo}}_B = 3, u^{\mathrm{exo}}_E = 4, u_1^* = 0.05, u_2^* = 0.05$.

### Step 2 — Action

Apply $\mathrm{do}(\neg m_1)$. Mechanism $m_1$ is removed; $C$ and $D$ become exogenous with point mass $0$ (the fallback $P_0$).

### Step 3 — Prediction

The abducted $u_1^*$ is now *unused* (its mechanism has been deleted). Carry $u_2^*$:

$$
C^{cf} = 0, \quad D^{cf} = 0, \quad F^{cf} = k_2 \cdot 0 \cdot 4 + u_2^* = 0.05
$$

This counterfactual query — **"what would the downstream product $F$ have been if we had ablated the upstream reaction?"** — has no Pearl-side single-intervention counterfactual. It requires the hypergraph framework to be expressible at all.

---

## 7. Irreducibility argument

**Claim.** Let $\mathcal{F}_{\mathrm{HG}}$ be the family of joint distributions over $(A, B, C, D)$ obtainable from any Hypergraph SCM with the incidence pattern of §1. Let $\mathcal{F}_{\mathrm{Pearl}}$ be the family obtainable from any Pearl SCM with single-output equations $C := f_C(A, B, u_C)$, $D := f_D(A, B, u_D)$ and $u_C \perp u_D$, both *non-trivial* (i.e., $u_C, u_D$ have nonzero variance).

Then $\mathcal{F}_{\mathrm{HG}} \not\subseteq \mathcal{F}_{\mathrm{Pearl}}$: there exists a distribution in $\mathcal{F}_{\mathrm{HG}}$ that no element of $\mathcal{F}_{\mathrm{Pearl}}$ can match.

**Proof sketch.** Take the Hypergraph SCM of §2. It induces $\Pr(C = D) = 1$. For any Pearl SCM in $\mathcal{F}_{\mathrm{Pearl}}$,

$$
\Pr(C = D) = \int \mathbb{1}[f_C(a, b, u_C) = f_D(a, b, u_D)] \, dP(a, b, u_C, u_D) < 1
$$

generically, because the integrand is supported on a measure-zero set in $(u_C, u_D)$-space whenever $f_C(a, b, \cdot)$ and $f_D(a, b, \cdot)$ are non-constant. □

**Resolution.** Pearl SCMs *with* latents are universal — adding a latent $L$ with $L \to C$ and $L \to D$ recovers the joint coupling. **The latent is the mechanism**: in our framework, $m_1$ *is* this latent, but named, intervenable, and structurally typed. The hypergraph framework's contribution is therefore not new expressive power but **first-class addressability of mechanisms**, which gives access to $\mathrm{do}(\neg m_1)$ and $\mathrm{do}(m_1 \to m_1')$ as well-defined single interventions.

---

## 8. d-separation, illustrated

The bipartite blowup $B(\mathcal{M})$:

```
A ──┐
    ├──> [m1] ──> C ──┐
B ──┘           ├──> [m2] ──> F
                D     │
                E ────┘
```

(In this diagram `[mₖ]` are mechanism nodes; arrows are the bipartite incidence.)

Under T1 (target theorem), Pearl d-separation on $B(\mathcal{M})$ governs hypergraph independence. For instance:

- $A \not\perp C \mid \emptyset$ (path $A \to m_1 \to C$ unblocked).
- $A \perp E \mid \emptyset$ (no undirected path).
- $A \not\perp D \mid \emptyset$ (path $A \to m_1 \to D$).
- $C \perp D \mid m_1$ — but $m_1$ is *not* a variable! In the hypergraph framework, conditioning on a mechanism is conditioning on the noise $u_1$, which is exogenous and unobservable. So $C$ and $D$ are *deterministically equal* but cannot be d-separated by conditioning on observable variables. This is a real subtlety: the deterministic-relations augmentation of d-separation (Geiger-Pearl 1990) is required to handle joint outputs correctly.

---

## 9. Tying to code

Every numerical claim in this document is reproduced by `minimal_model/`:

- `examples.reaction_network()` builds the SCM of §2.
- `test_example.py::test_observational` checks $\mathrm{Corr}(C, D) = 1$ on samples.
- `test_example.py::test_intervention_1` verifies $\mathrm{do}(A = 4)$.
- `test_example.py::test_intervention_2` verifies $\mathrm{do}(\neg m_1) \Rightarrow C = D = 0$.
- `test_example.py::test_intervention_3` verifies $\mathrm{do}(m_1 \to m_1')$ preserves $C = D$.
- `test_example.py::test_counterfactual` reproduces the §5 hand calculation to 6 decimal places.
- `test_example.py::test_counterfactual_mechanism_deletion` reproduces §6.
- `test_example.py::test_irreducibility_witness` exhibits the $\mathrm{Corr}(C, D) = 1$ vs. Pearl-decomposed comparison.
