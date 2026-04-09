import unittest

import torch

from losses.losses import Grad as GradGPU
from losses.losses_cpu import Grad as GradCPU


class MaskAwareGradNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.flow = torch.tensor(
            [[
                [[0.0, 1.0, 3.0],
                 [1.0, 2.0, 4.0],
                 [3.0, 5.0, 8.0]],
                [[2.0, 4.0, 7.0],
                 [3.0, 6.0, 8.0],
                 [4.0, 8.0, 11.0]],
            ]],
            dtype=torch.float32,
        )
        self.full_mask = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        self.partial_mask = torch.tensor(
            [[
                [[1.0, 1.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0]]
            ]],
            dtype=torch.float32,
        )

    def _expected_masked_grad(self, penalty, mask):
        dy = self.flow[:, :, 1:, :] - self.flow[:, :, :-1, :]
        dx = self.flow[:, :, :, 1:] - self.flow[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        transform = torch.pow if penalty == "l2" else torch.abs
        if penalty == "l2":
            dy_values = transform(dy, 2)
            dx_values = transform(dx, 2)
        else:
            dy_values = transform(dy)
            dx_values = transform(dx)

        num_channels = self.flow.shape[1]
        dy_mean = (dy_values * mask_dy).sum() / (mask_dy.sum() * num_channels)
        dx_mean = (dx_values * mask_dx).sum() / (mask_dx.sum() * num_channels)
        return (dy_mean + dx_mean) / 2.0

    def test_full_mask_matches_unmasked_mean(self):
        for grad_cls in (GradGPU, GradCPU):
            with self.subTest(implementation=grad_cls.__module__):
                grad = grad_cls(penalty="l2")
                masked = grad(self.flow, mask=self.full_mask)
                unmasked = grad(self.flow)
                self.assertTrue(torch.isclose(masked, unmasked, atol=1e-6))

    def test_partial_mask_averages_over_channels_once(self):
        for grad_cls in (GradGPU, GradCPU):
            for penalty in ("l1", "l2"):
                with self.subTest(implementation=grad_cls.__module__, penalty=penalty):
                    grad = grad_cls(penalty=penalty)
                    masked = grad(self.flow, mask=self.partial_mask)
                    expected = self._expected_masked_grad(penalty, self.partial_mask)
                    self.assertTrue(torch.isclose(masked, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
