"""A differential oracle for Pearl-ADMG identification.

The conformance generator cannot test an ID algorithm. It hides at most one variable and
never an exogenous one, so the graphs it produces reach only the first two lines of the
seven-line recursion -- lines 3, 5 and 7 never fire, and it has never once produced a
hedge. A sweep over it would report green while most of the algorithm went unexecuted.

So the corpus is built from the algorithm's own structure and from the literature's
published examples, and the oracle is generic: given an ADMG, `random_scm` builds a
concrete structural model compatible with it -- one latent per bidirected edge, one noise
per node -- from which both `P(V)` and `P(Y | do(X))` are computed exactly by enumeration.
An identifying formula is then checked against the second using only the first.

That is a real oracle rather than a restatement: nothing in it knows how the estimand was
derived, and a formula that is merely plausible fails it.
"""
from .oracle import ADMGModel, random_scm

__all__ = ["ADMGModel", "random_scm"]
