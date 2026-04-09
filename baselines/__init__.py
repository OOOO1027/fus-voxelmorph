"""
Traditional registration methods for comparison.

Baseline methods:
- Rigid: Rigid body registration (rotation + translation)
- Affine: Affine transformation (rigid + scaling + shearing)
- Demons: Optical flow-based deformable registration
- BSpline: Free-form deformation using B-splines

Dependencies:
- SimpleITK: for all traditional methods
- OpenCV: optional, for additional rigid/affine methods
"""

from .traditional import (
    RigidRegistration,
    AffineRegistration,
    DemonsRegistration,
    BSplineRegistration,
)

from .comparison import (
    compare_methods,
    run_comparison_experiment,
    create_comparison_table,
    create_comparison_figures,
)

__all__ = [
    'RigidRegistration',
    'AffineRegistration',
    'DemonsRegistration',
    'BSplineRegistration',
    'compare_methods',
    'run_comparison_experiment',
    'create_comparison_table',
    'create_comparison_figures',
]
