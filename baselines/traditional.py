"""
Traditional image registration methods using SimpleITK.

Implements:
1. Rigid registration (rotation + translation)
2. Affine registration (rigid + scaling + shearing)
3. Demons deformable registration
4. B-spline deformable registration
"""

import time
import numpy as np

# Check for SimpleITK
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("Warning: SimpleITK not installed. Traditional methods will not work.")
    print("Install with: pip install SimpleITK")


class BaseRegistration:
    """Base class for traditional registration methods."""

    def __init__(self, metric='mean_squares', optimizer='gradient_descent'):
        """
        Parameters
        ----------
        metric : str
            Similarity metric: 'mean_squares', 'normalized_correlation', 'mutual_information'
        optimizer : str
            Optimizer: 'gradient_descent', 'amoeba', 'powell'
        """
        self.metric = metric
        self.optimizer = optimizer
        self.execution_time = None
        self.transform = None

    def _get_metric(self):
        """Get SimpleITK metric."""
        if not HAS_SITK:
            raise ImportError("SimpleITK is required")

        metric_map = {
            'mean_squares': sitk.MeanSquaresImageToImageMetricv4(),
            'normalized_correlation': sitk.CorrelationImageToImageMetricv4(),
            'mutual_information': sitk.MattesMutualInformationImageToImageMetricv4(),
            'mattes_mi': sitk.MattesMutualInformationImageToImageMetricv4(),
        }

        if self.metric not in metric_map:
            raise ValueError(f"Unknown metric: {self.metric}")

        return metric_map[self.metric]

    def _numpy_to_sitk(self, image):
        """Convert numpy array to SimpleITK image."""
        if not HAS_SITK:
            raise ImportError("SimpleITK is required")

        # Ensure 2D
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        # Convert to float32
        image = image.astype(np.float32)

        sitk_image = sitk.GetImageFromArray(image)
        return sitk_image

    def _sitk_to_numpy(self, sitk_image):
        """Convert SimpleITK image to numpy array."""
        return sitk.GetArrayFromImage(sitk_image)

    def register(self, moving, fixed):
        """
        Register moving image to fixed image.

        Parameters
        ----------
        moving : np.ndarray (H, W)
            Moving image
        fixed : np.ndarray (H, W)
            Fixed image

        Returns
        -------
        warped : np.ndarray (H, W)
            Warped moving image
        transform_params : dict
            Transformation parameters
        """
        raise NotImplementedError

    def warp(self, moving, transform_params):
        """
        Apply transformation to moving image.

        Parameters
        ----------
        moving : np.ndarray (H, W)
        transform_params : dict

        Returns
        -------
        warped : np.ndarray (H, W)
        """
        raise NotImplementedError

    def get_displacement_field(self, shape):
        """
        Get displacement field as (2, H, W) array.

        Parameters
        ----------
        shape : tuple
            Output shape (H, W)

        Returns
        -------
        flow : np.ndarray (2, H, W)
            Displacement field [dx, dy]
        """
        raise NotImplementedError


class RigidRegistration(BaseRegistration):
    """
    Rigid body registration using Euler2D transform.

    Transforms: rotation + translation (3 DOF)
    """

    def __init__(self, metric='mean_squares', optimizer='gradient_descent',
                 max_iterations=200, learning_rate=1.0):
        super().__init__(metric, optimizer)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def register(self, moving, fixed):
        """Rigid registration."""
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for RigidRegistration")

        # Convert to SimpleITK
        moving_sitk = self._numpy_to_sitk(moving)
        fixed_sitk = self._numpy_to_sitk(fixed)

        # Initial transform
        initial_transform = sitk.Euler2DTransform(
            sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        )

        # Registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Metric
        if self.metric == 'mean_squares':
            registration_method.SetMetricAsMeanSquares()
        elif self.metric == 'normalized_correlation':
            registration_method.SetMetricAsCorrelation()
        elif self.metric in ['mutual_information', 'mattes_mi']:
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # Optimizer
        if self.optimizer == 'gradient_descent':
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=self.learning_rate,
                numberOfIterations=self.max_iterations,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10
            )
        elif self.optimizer == 'amoeba':
            registration_method.SetOptimizerAsAmoeba(
                simplexDelta=0.1,
                numberOfIterations=self.max_iterations
            )

        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Execute
        start_time = time.time()
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        self.execution_time = time.time() - start_time

        self.transform = final_transform

        # Warp moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)

        warped_sitk = resampler.Execute(moving_sitk)
        warped = self._sitk_to_numpy(warped_sitk)

        # Transform parameters
        transform_params = {
            'type': 'rigid',
            'angle': final_transform.GetAngle(),
            'translation': final_transform.GetTranslation(),
            'center': final_transform.GetCenter(),
            'matrix': final_transform.GetMatrix(),
            'time': self.execution_time,
        }

        return warped, transform_params

    def get_displacement_field(self, shape):
        """Get displacement field from rigid transform."""
        if self.transform is None:
            raise RuntimeError("Must call register() first")

        H, W = shape

        # Create identity transform for comparison
        identity = sitk.Transform(2, sitk.sitkIdentity)

        # Create displacement field
        displacement = sitk.TransformToDisplacementField(
            self.transform,
            sitk.sitkVectorFloat64,
            size=[W, H],
            outputOrigin=[0, 0],
            outputSpacing=[1.0, 1.0],
            outputDirection=[1, 0, 0, 1]
        )

        # Convert to numpy
        disp_array = self._sitk_to_numpy(displacement)

        # SimpleITK returns (dx, dy) as last dimension
        flow = np.transpose(disp_array, (2, 0, 1))  # (2, H, W)

        return flow.astype(np.float32)


class AffineRegistration(BaseRegistration):
    """
    Affine registration using AffineTransform.

    Transforms: rotation + translation + scaling + shearing (6 DOF)
    """

    def __init__(self, metric='mean_squares', optimizer='gradient_descent',
                 max_iterations=200, learning_rate=1.0):
        super().__init__(metric, optimizer)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def register(self, moving, fixed):
        """Affine registration."""
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for AffineRegistration")

        # Convert to SimpleITK
        moving_sitk = self._numpy_to_sitk(moving)
        fixed_sitk = self._numpy_to_sitk(fixed)

        # Initial transform
        initial_transform = sitk.AffineTransform(
            sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.AffineTransform(2),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        )

        # Registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Metric
        if self.metric == 'mean_squares':
            registration_method.SetMetricAsMeanSquares()
        elif self.metric == 'normalized_correlation':
            registration_method.SetMetricAsCorrelation()
        elif self.metric in ['mutual_information', 'mattes_mi']:
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # Optimizer
        if self.optimizer == 'gradient_descent':
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=self.learning_rate,
                numberOfIterations=self.max_iterations,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10
            )
        elif self.optimizer == 'amoeba':
            registration_method.SetOptimizerAsAmoeba(
                simplexDelta=0.1,
                numberOfIterations=self.max_iterations
            )

        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Execute
        start_time = time.time()
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        self.execution_time = time.time() - start_time

        self.transform = final_transform

        # Warp moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)

        warped_sitk = resampler.Execute(moving_sitk)
        warped = self._sitk_to_numpy(warped_sitk)

        # Transform parameters
        transform_params = {
            'type': 'affine',
            'matrix': final_transform.GetMatrix(),
            'translation': final_transform.GetTranslation(),
            'center': final_transform.GetCenter(),
            'time': self.execution_time,
        }

        return warped, transform_params

    def get_displacement_field(self, shape):
        """Get displacement field from affine transform."""
        if self.transform is None:
            raise RuntimeError("Must call register() first")

        H, W = shape

        # Create displacement field
        displacement = sitk.TransformToDisplacementField(
            self.transform,
            sitk.sitkVectorFloat64,
            size=[W, H],
            outputOrigin=[0, 0],
            outputSpacing=[1.0, 1.0],
            outputDirection=[1, 0, 0, 1]
        )

        # Convert to numpy
        disp_array = self._sitk_to_numpy(displacement)

        # SimpleITK returns (dx, dy) as last dimension
        flow = np.transpose(disp_array, (2, 0, 1))  # (2, H, W)

        return flow.astype(np.float32)


class DemonsRegistration(BaseRegistration):
    """
    Demons deformable registration algorithm.

    Optical flow-based deformable registration.
    Fast but may produce less smooth deformations.
    """

    def __init__(self, iterations=100, smooth_sigma=1.0):
        """
        Parameters
        ----------
        iterations : int
            Number of iterations
        smooth_sigma : float
            Gaussian smoothing sigma for regularization
        """
        super().__init__()
        self.iterations = iterations
        self.smooth_sigma = smooth_sigma
        self.displacement_field = None

    def register(self, moving, fixed):
        """Demons registration."""
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for DemonsRegistration")

        # Convert to SimpleITK
        moving_sitk = self._numpy_to_sitk(moving)
        fixed_sitk = self._numpy_to_sitk(fixed)

        # Demons registration
        demons_filter = sitk.DemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(self.iterations)
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(self.smooth_sigma)

        # Execute
        start_time = time.time()
        displacement_field = demons_filter.Execute(fixed_sitk, moving_sitk)
        self.execution_time = time.time() - start_time

        self.displacement_field = displacement_field

        # Apply displacement field
        warper = sitk.WarpImageFilter()
        warper.SetOutputParameterMapFromImage(fixed_sitk)
        warped_sitk = warper.Execute(moving_sitk, displacement_field)
        warped = self._sitk_to_numpy(warped_sitk)

        transform_params = {
            'type': 'demons',
            'iterations': self.iterations,
            'smooth_sigma': self.smooth_sigma,
            'time': self.execution_time,
        }

        return warped, transform_params

    def get_displacement_field(self, shape=None):
        """Get displacement field."""
        if self.displacement_field is None:
            raise RuntimeError("Must call register() first")

        # Convert to numpy
        disp_array = self._sitk_to_numpy(self.displacement_field)

        # SimpleITK returns (dx, dy) as last dimension
        flow = np.transpose(disp_array, (2, 0, 1))  # (2, H, W)

        return flow.astype(np.float32)


class BSplineRegistration(BaseRegistration):
    """
    B-spline deformable registration.

    Free-form deformation using B-spline control points.
    Produces smooth deformations.
    """

    def __init__(self, grid_size=(10, 10), iterations=100,
                 metric='mean_squares', optimizer='lbfgs'):
        """
        Parameters
        ----------
        grid_size : tuple
            Number of B-spline control points (nx, ny)
        iterations : int
            Number of optimizer iterations
        metric : str
            Similarity metric
        optimizer : str
            Optimizer: 'lbfgs', 'gradient_descent'
        """
        super().__init__(metric, optimizer)
        self.grid_size = grid_size
        self.iterations = iterations
        self.transform = None

    def register(self, moving, fixed):
        """B-spline registration."""
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for BSplineRegistration")

        # Convert to SimpleITK
        moving_sitk = self._numpy_to_sitk(moving)
        fixed_sitk = self._numpy_to_sitk(fixed)

        # Get image dimensions
        size = fixed_sitk.GetSize()

        # Transform domain mesh size (number of control points - spline order)
        transform_mesh_size = [self.grid_size[0] - 3, self.grid_size[1] - 3]

        # Create B-spline transform
        bspline_transform = sitk.BSplineTransformInitializer(
            fixed_sitk,
            transform_mesh_size,
            order=3
        )

        # Registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Metric
        if self.metric == 'mean_squares':
            registration_method.SetMetricAsMeanSquares()
        elif self.metric == 'normalized_correlation':
            registration_method.SetMetricAsCorrelation()
        elif self.metric in ['mutual_information', 'mattes_mi']:
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # Optimizer
        if self.optimizer == 'lbfgs':
            registration_method.SetOptimizerAsLBFGS2(
                solutionAccuracy=1e-5,
                numberOfIterations=self.iterations
            )
        elif self.optimizer == 'gradient_descent':
            registration_method.SetOptimizerAsGradientDescentLineSearch(
                learningRate=1.0,
                numberOfIterations=self.iterations
            )

        registration_method.SetInitialTransform(bspline_transform, inPlace=True)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Shrink factors and smoothing sigmas for multi-resolution
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

        # Execute
        start_time = time.time()
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        self.execution_time = time.time() - start_time

        self.transform = final_transform

        # Warp moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)

        warped_sitk = resampler.Execute(moving_sitk)
        warped = self._sitk_to_numpy(warped_sitk)

        transform_params = {
            'type': 'bspline',
            'grid_size': self.grid_size,
            'iterations': self.iterations,
            'time': self.execution_time,
        }

        return warped, transform_params

    def get_displacement_field(self, shape):
        """Get displacement field from B-spline transform."""
        if self.transform is None:
            raise RuntimeError("Must call register() first")

        H, W = shape

        # Create displacement field
        displacement = sitk.TransformToDisplacementField(
            self.transform,
            sitk.sitkVectorFloat64,
            size=[W, H],
            outputOrigin=[0, 0],
            outputSpacing=[1.0, 1.0],
            outputDirection=[1, 0, 0, 1]
        )

        # Convert to numpy
        disp_array = self._sitk_to_numpy(displacement)

        # SimpleITK returns (dx, dy) as last dimension
        flow = np.transpose(disp_array, (2, 0, 1))  # (2, H, W)

        return flow.astype(np.float32)


class OpenCVRigidRegistration(BaseRegistration):
    """
    Rigid registration using OpenCV (alternative implementation).
    Uses ECC (Enhanced Correlation Coefficient) maximization.
    """

    def __init__(self, motion_type='euclidean', max_iterations=100):
        """
        Parameters
        ----------
        motion_type : str
            'translation', 'euclidean', 'affine', 'homography'
        max_iterations : int
            Maximum iterations
        """
        super().__init__()
        self.motion_type = motion_type
        self.max_iterations = max_iterations
        self.warp_matrix = None

        # Check OpenCV
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV is required for OpenCVRigidRegistration")

    def register(self, moving, fixed):
        """OpenCV-based rigid registration."""
        cv2 = self.cv2

        # Convert to float32
        moving_float = moving.astype(np.float32)
        fixed_float = fixed.astype(np.float32)

        # Define motion model
        motion_map = {
            'translation': cv2.MOTION_TRANSLATION,
            'euclidean': cv2.MOTION_EUCLIDEAN,
            'affine': cv2.MOTION_AFFINE,
            'homography': cv2.MOTION_HOMOGRAPHY,
        }

        warp_mode = motion_map.get(self.motion_type, cv2.MOTION_EUCLIDEAN)

        # Initialize warp matrix
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                   self.max_iterations, 1e-6)

        # Run ECC algorithm
        start_time = time.time()
        try:
            cc, warp_matrix = cv2.findTransformECC(
                fixed_float, moving_float, warp_matrix,
                warp_mode, criteria
            )
        except cv2.error as e:
            print(f"OpenCV ECC failed: {e}")
            # Fallback to identity
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            cc = 0

        self.execution_time = time.time() - start_time
        self.warp_matrix = warp_matrix

        # Warp image
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warped = cv2.warpPerspective(moving_float, warp_matrix,
                                         (fixed.shape[1], fixed.shape[0]),
                                         flags=cv2.INTER_LINEAR)
        else:
            warped = cv2.warpAffine(moving_float, warp_matrix,
                                   (fixed.shape[1], fixed.shape[0]),
                                   flags=cv2.INTER_LINEAR)

        transform_params = {
            'type': f'opencv_{self.motion_type}',
            'warp_matrix': warp_matrix,
            'correlation': cc,
            'time': self.execution_time,
        }

        return warped, transform_params

    def get_displacement_field(self, shape):
        """Get displacement field from warp matrix."""
        if self.warp_matrix is None:
            raise RuntimeError("Must call register() first")

        H, W = shape
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)

        # Apply warp matrix to coordinates
        if self.warp_matrix.shape == (2, 3):  # Affine
            # [x', y'] = M * [x, y, 1]
            x_warped = self.warp_matrix[0, 0] * x + self.warp_matrix[0, 1] * y + self.warp_matrix[0, 2]
            y_warped = self.warp_matrix[1, 0] * x + self.warp_matrix[1, 1] * y + self.warp_matrix[1, 2]
        elif self.warp_matrix.shape == (3, 3):  # Homography
            # [x', y', w] = M * [x, y, 1]
            w = self.warp_matrix[2, 0] * x + self.warp_matrix[2, 1] * y + self.warp_matrix[2, 2]
            x_warped = (self.warp_matrix[0, 0] * x + self.warp_matrix[0, 1] * y + self.warp_matrix[0, 2]) / w
            y_warped = (self.warp_matrix[1, 0] * x + self.warp_matrix[1, 1] * y + self.warp_matrix[1, 2]) / w
        else:
            raise ValueError(f"Unknown warp matrix shape: {self.warp_matrix.shape}")

        # Displacement = warped - original
        flow_x = x_warped - x
        flow_y = y_warped - y

        flow = np.stack([flow_x, flow_y], axis=0)
        return flow.astype(np.float32)
