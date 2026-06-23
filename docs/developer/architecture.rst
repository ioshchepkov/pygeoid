Architecture
============

Overview
--------

The main architectural goal is to separate and generalize reusable computational
machinery from narrow domain-specific geodetic routines. This is a conscious decision:
many mathematical and computational methods encountered across different
geodetic domains turn out, on closer inspection, to share the same abstract
structure, even when their physical interpretation, input data, or conventional
terminology differ.

If successful, this should keep narrow domain packages compact
both internally and from the user's perspective, leaving them focused on
geodetic meaning and physical interpretation rather than duplicated
computational machinery.

Design Principles
-----------------

PyGeoid should follow these design principles:

* **Open and reproducible research software.** Follow best practices in open
  science and research software development, including FAIR4RS where applicable,
  and help users apply them in their own work.

* **Explicit is better than implicit.** Units, conventions, reference frames,
  and uncertainties should be explicit throughout computational pipelines and
  centralized where possible.

* **DRY and methodological pluralism.** Shared computational machinery should
  not be duplicated across domains, but alternative scientifically justified
  methods for computing the same quantity should be encouraged rather than
  treated as duplication.

* **Validated, testable, and transparent methods.** Important public methods
  should be documented with formulas, assumptions, references, and validation
  evidence where practical; experimental methods may start lighter.

* **Interoperable but lightweight core.** The package should work naturally with
  the scientific Python and geospatial ecosystem, while keeping the core
  dependency set small and making heavy or specialized dependencies optional.

* **Performance with scientific clarity.** Efficient implementations are
  encouraged, especially for large grids, high-degree spectral models, and
  time-variable fields, but performance should not hide formulas, conventions,
  assumptions, or reference methods.


Core Computational Model
------------------------

PyGeoid uses the mathematical and physical concept of a field as its main
computational abstraction.

The guiding idea is that, in geodetic computation, almost everything important
is either a field, an operator on fields, a functional of fields, or an
observable derived from them. These concepts form bidirectional modelling
framework::

    Field <-> Operator <-> Functional <-> Observables

In the forward direction, fields are transformed by operators, interpreted
through physical or geodetic functionals, and connected to observable
quantities.

In practical data pipelines, the direction is often reversed: observations are
collected first and then used to estimate, infer, or constrain the underlying
fields and model parameters, from which geodetic functionals are derived.

This makes the same conceptual structure useful for both forward modelling and
inverse problems. The complexity of geodetic workflows means that direct and
inverse problems may appear several times within the same pipeline, involving
different underlying fields, their derivatives, and derived functionals.

Top-Level Structure
-------------------

The intended top-level package structure is::

    pygeoid/
    ├── conventions/
    ├── geometry/
    ├── numerics/
    ├── fields/
    ├── uncertainty/
    ├── earth/
    ├── reference/
    ├── observations/
    ├── estimation/
    └── io/
