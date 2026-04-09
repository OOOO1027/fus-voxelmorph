# Baseline Registration Methods

Traditional image registration methods for comparison with VoxelMorph.

## Requirements

```bash
pip install SimpleITK
```

## Methods

### RigidRegistration
- 3 DOF: rotation + translation
- Best for: pure rigid motion

### AffineRegistration  
- 6 DOF: rotation + translation + scale + shear
- Best for: global linear deformation

### DemonsRegistration
- Optical flow-based deformable registration
- Best for: smooth, small deformations
- Fast but sensitive to noise

### BSplineRegistration
- Free-form deformation with B-splines
- Best for: complex local deformations
- Produces smooth deformations

## Usage

```python
from baselines import RigidRegistration, compare_methods

# Create method
rigid = RigidRegistration(metric='mean_squares', max_iterations=200)

# Register
warped, params = rigid.register(source, target)

# Get displacement field
flow = rigid.get_displacement_field(source.shape)

# Compare multiple methods
methods = {'Rigid': rigid, 'VoxelMorph': model}
results = compare_methods(source, target, methods, device='cuda')
```

## Run Comparison

```bash
# Full comparison
python compare_baseline.py --synthetic --n_samples 20

# With trained model
python compare_baseline.py \
    --data_path data/test/ \
    --model checkpoints/best_model.pth \
    --output_dir comparison/
```
