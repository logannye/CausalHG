# Pre-flight: which causal questions can this screen actually answer?

```
python3 demo/preflight.py
```

Runs in about two seconds, from a clean checkout, with no third-party packages. The data is
vendored under `data/`.

## What this is

A perturbation screen is expensive and its analysis is cheap, so the analysis is where the
mistakes go unnoticed. This demo points the compiler at a real published screen — Norman et
al. 2019, 65,359 K562 cells, 105 single-gene arms plus a non-targeting control — and asks a
grid of mechanism-level causal questions: which of these can the data answer?

Most cannot. **Each verdict is printed next to evidence assembled independently of the
compiler**, because a tool that says "no" is only worth having if the "no" is right.

| # | question | verdict | why it is right |
|---|---|---|---|
| 1 | 95 mechanism-deletion queries over the curated pathway | **89 refused**, 6 identified, 0 of them non-trivial | 12 of 20 mechanisms lie on a cycle; for a mechanism on a cycle the observational conditional is not its structural kernel |
| 2 | adjust the KLF1 → HBG1 effect for total UMI and cell-cycle score | **refused** — both post-treatment | both are measured on the same cell *after* it was perturbed, so they are descendants of the intervention. Structural, not an assumption |
| 3 | what does activating KLF1 do to SLC4A1? | **answered, resting on 19 of 11,183 cells** | SLC4A1 is detected in 0.8% of K562 cells. Positivity *passes* — the cells are scarce, not absent — so no refusal catches this and only the disclosed stratum count does |
| 4 | three expression levels per gene, declared the obvious way | **refused** — a third of the scope undefined | the median non-zero UMI count is 1, so the middle level is unreachable while the domain still claims it |
| 5 | the estimate is reported against 11,183 control cells — is it? | **3× overstatement**, 3,734 effective rows | the estimand re-weights the data by the policy, so the answer rests on the rows at the levels the policy leans on |

Three refusals and two disclosures. The disclosures matter as much: verdicts 3 and 5 both
*return a number*, and both would read as clean results without the certificate beside them.

## The constructive half

A refusal that does not say what would work is a complaint, not an instrument. The demo ends
by measuring what design would answer these questions.

A cycle is a statement about variables measured at one instant. Resolved in time it is a
DAG, because `gene_t(k+1)` depends on its regulators at `t(k)` — and **no edge is removed**,
which matters, because pruning a network until it is acyclic would manufacture the very
property the refusal is about.

| timepoints | variables | identified | non-trivial | max induced width |
|---|---|---|---|---|
| 1 (as curated) | 20 | 6 | **0** | — |
| 2 | 40 | 100 | 0 | — |
| 3 | 60 | 100 | **35** | 8 |
| 4 | 80 | 100 | **69** | 12 |

Time-resolution, not more cells. That is a specification an experiment can be designed
against, and it is what a pre-flight check is for.

## "Non-trivial" is a measured property, not a rhetorical one

A deletion estimand with **one** estimated conditional factor and a point-mass policy is
`P(Y | X = x)` — a group-by, agreeing with `pandas` bit for bit at every sample size tested.
At **two or more** factors it is a genuinely different, Markov-restricted estimator. The
demo counts factors rather than asserting importance, and reports 0 non-trivial answers on
the curated graph for that reason.

## Data provenance

`data/norman_cells.tsv.gz` — 65,359 cells × 23 genes, plus arm, gemgroup and total UMI.
Derived from GEO accession **GSE133344** (Norman, Horlbeck, Replogle et al., "Exploring
genetic interaction manifolds constructed from rich single-cell phenotypes", *Science*
365(6455), 2019). Cells are kept when their guide identity resolves to a single perturbed
gene or to a non-targeting control; two-gene arms are dropped because they pose a different
query shape this demo does not ask.

`data/collectri_edges.tsv` — 75 directed edges among those genes, from **CollecTRI**
(Müller-Dott et al., 2023), carrying the constituent-resource tags. The tags are kept
because the cycle finding rests on them: these cycles survive restriction to TRRUST alone
(253-node strongly connected component) and to DoRothEA-A alone (103-node SCC), so they are
not an artefact of aggregating databases with different context coverage.

Regenerate both with:

```
python3 demo/build_cache.py --norman-dir DIR --collectri PATH
```

which streams the 362M-entry matrix once. Neither source file is redistributed here; only
the derived counts for the named genes.

## What this demo does not claim

It does **not** show that the compiler predicts a held-out interventional arm. It was tried,
on the CEBPA arm, and it lost to "nothing changed" — the diagnosis is in
`src/causal_hypergraphs/estimation/estimator.py` under `PolicySupport`: CRISPRa drives its
target 24.8× past anything the control cells show, so a kernel fitted on control data
answers with the observational noise-slope. That is off-support extrapolation, and no
identification machinery repairs it. Verdict 5 exists because of it.
