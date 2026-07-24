# Risch literature

This directory collects papers and books used by the implementation roadmap.
The papers were downloaded from publisher free-access pages, author or
university pages, institutional repositories, and arXiv on 2026-07-24. The
collection in `books/` was supplied directly by the user.

## Available locally

### Consolidated source collection

[`books/integration-in-finite-terms-fundamental-sources-2022.pdf`](books/integration-in-finite-terms-fundamental-sources-2022.pdf)
is Clemens G. Raab and Michael F. Singer, editors, *Integration in Finite
Terms: Fundamental Sources* (Springer, 2022),
[DOI 10.1007/978-3-030-98767-1](https://doi.org/10.1007/978-3-030-98767-1).
The user supplied this copy directly. Its 303 PDF pages include:

| PDF page | Work |
| ---: | --- |
| 9 | Maxwell Rosenlicht, "Integration in Finite Terms" |
| 19 | Michael F. Singer, comments on Rosenlicht's paper |
| 39 | J. F. Ritt, *Integration in Finite Terms: Liouville's Theory of Elementary Methods* |
| 143 | Askold Khovanskii, comments on Ritt's book |
| 208 | Robert H. Risch, "On the Integration of Elementary Functions which are Built Up Using Algebraic Operations" |
| 225 | Clemens G. Raab, comments on Risch's paper |
| 238 | Barry M. Trager, *Integration of Algebraic Functions* |
| 295 | Barry M. Trager, retrospective comments on algebraic integration |

This is the preferred consolidated source for Liouville descent, the transition
from transcendental towers to arbitrary algebraic operations, and the
integral-basis/divisor algorithms required to complete the algebraic branch.
The separate Trager PDF below remains useful because it is smaller and has the
thesis's original page numbering.

### Individual papers

| File | Reference | Source | Why it matters |
| --- | --- | --- | --- |
| [`papers/risch-1969-problem-of-integration-in-finite-terms.pdf`](papers/risch-1969-problem-of-integration-in-finite-terms.pdf) | Robert H. Risch, "The Problem of Integration in Finite Terms," *Transactions of the AMS* 139 (1969), 167-189 | [AMS free digital archive](https://www.ams.org/journals/tran/1969-139-00/S0002-9947-1969-0237477-8/) | Foundational structure theorem and decision problem |
| [`papers/bronstein-1998-symbolic-integration-tutorial.pdf`](papers/bronstein-1998-symbolic-integration-tutorial.pdf) | Manuel Bronstein, "Symbolic Integration Tutorial," ISSAC 1998 | [Author's INRIA page](https://www-sop.inria.fr/cafe/Manuel.Bronstein/publications/issac98.pdf) | Compact overview of the full algorithm |
| [`papers/bronstein-1998-lazy-hermite-reduction.pdf`](papers/bronstein-1998-lazy-hermite-reduction.pdf) | Manuel Bronstein, "The Lazy Hermite Reduction," INRIA RR-3562 (1998) | [HAL/INRIA](https://inria.hal.science/inria-00073121) | Lazy integral-basis refinement |
| [`papers/trager-1984-integration-of-algebraic-functions.pdf`](papers/trager-1984-integration-of-algebraic-functions.pdf) | Barry M. Trager, *Integration of Algebraic Functions*, PhD thesis, MIT (1984) | [University of Pisa course archive](https://people.dm.unipi.it/gianni/Alg-Comp/thesis.pdf) | General algebraic integration, integral bases, divisors, and logarithmic terms |
| [`papers/davenport-1984-integration-on-an-algebraic-curve.pdf`](papers/davenport-1984-integration-on-an-algebraic-curve.pdf) | J. H. Davenport, "Intégration algorithmique des fonctions élémentairement transcendantes sur une courbe algébrique," *Annales de l'Institut Fourier* 34 (1984), 271-276 | [Numdam](https://www.numdam.org/item/AIF_1984__34_2_271_0/) | Transcendental extensions over algebraic curves |
| [`papers/bronstein-1991-risch-differential-equation-algebraic-curve.pdf`](papers/bronstein-1991-risch-differential-equation-algebraic-curve.pdf) | Manuel Bronstein, "The Risch Differential Equation on an Algebraic Curve," ISSAC 1991, 241-246 | [ACM free access](https://dl.acm.org/doi/10.1145/120694.120731) | Algebraic-curve Risch differential equations |
| [`papers/bronstein-1989-simplification-of-real-elementary-functions.pdf`](papers/bronstein-1989-simplification-of-real-elementary-functions.pdf) | Manuel Bronstein, "Simplification of Real Elementary Functions," ISSAC 1989, 207-211 | [ACM free access](https://dl.acm.org/doi/10.1145/74540.74566) | Real structure theorem and real elementary relations |
| [`papers/corless-jeffrey-1996-unwinding-number.pdf`](papers/corless-jeffrey-1996-unwinding-number.pdf) | Robert M. Corless and David J. Jeffrey, "The Unwinding Number," *ACM SIGSAM Bulletin* 30.2 (1996), 28-35 | [ACM free access](https://dl.acm.org/doi/10.1145/235699.235705) | Branch-correct logarithmic and inverse-trigonometric identities |
| [`papers/liang-jeffrey-2008-parametric-minimization-and-continuous-trig-integrals.pdf`](papers/liang-jeffrey-2008-parametric-minimization-and-continuous-trig-integrals.pdf) | Songxin Liang and David J. Jeffrey, "Unconstrained Parametric Minimization of a Polynomial: Approximate and Exact," ASCM 2007 proceedings (2008), 22-31 | [Author's University of Western Ontario page](https://www.uwo.ca/apmaths/faculty/jeffrey/pdfs/50810022.pdf) | Rectifying transformations for continuous real trigonometric antiderivatives |
| [`papers/raab-2013-generalization-of-risch-algorithm-to-special-functions.pdf`](papers/raab-2013-generalization-of-risch-algorithm-to-special-functions.pdf) | Clemens G. Raab, "Generalization of Risch's Algorithm to Special Functions" (2013) | [arXiv:1305.1481](https://arxiv.org/abs/1305.1481) | Optional post-elementary extension track |
| [`papers/corless-et-al-2017-symbolic-numeric-integration-rational-functions.pdf`](papers/corless-et-al-2017-symbolic-numeric-integration-rational-functions.pdf) | Robert M. Corless et al., "Symbolic-Numeric Integration of Rational Functions" (2017) | [Author's University of Western Ontario page](https://www.csd.uwo.ca/~mmorenom/Publications/Symbolic-Numeric-Integration-of-Rational-Functions.pdf) | Modern explanation and testing reference for Lazard-Rioboo-Trager reduction |

Bronstein's *Symbolic Integration I: Transcendental Functions* is already
available on this machine as `/Users/nimish/Downloads/symbolic_1.pdf`.  It is
not duplicated here because it is a book rather than a paper and was supplied
separately by the user.

## Cited works not downloaded

| Reference | Public record | Status |
| --- | --- | --- |
| Manuel Bronstein, "Integration of Elementary Functions," *Journal of Symbolic Computation* 9 (1990), 117-173 | [DOI](https://doi.org/10.1016/S0747-7171(08)80027-2) | The publisher page advertises a PDF, but automated access reaches a robot challenge. No author or repository PDF was found. |
| Manuel Bronstein, "A Unification of Liouvillian Extensions," *Applicable Algebra in Engineering, Communication and Computing* 1 (1990), 5-24 | [IBM Research record](https://research.ibm.com/publications/a-unification-of-liouvillian-extensions) | Metadata and abstract are public; no authorized public PDF was found. |
| Daniel Lazard and Renaud Rioboo, "Integration of Rational Functions: Rational Computation of the Logarithmic Part," *Journal of Symbolic Computation* 9 (1990), 113-115 | [DOI](https://doi.org/10.1016/S0747-7171(08)80026-0) | No authorized public copy of the original paper was found. The 2017 Corless et al. paper above gives a detailed modern account of the algorithm. |
| J. H. Davenport, *On the Integration of Algebraic Functions*, LNCS 102 (1981) | [University of Bath record](https://researchportal.bath.ac.uk/en/publications/on-the-integration-of-algebraic-functions/) | The Springer book is not publicly downloadable. Trager's thesis and Davenport's 1984 paper cover much of the immediately relevant material. |

If an author or copyright holder supplies any missing PDF directly, place it in
`references/papers/` and update this index with its provenance.
