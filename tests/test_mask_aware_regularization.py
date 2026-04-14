import unittest

import torch

from losses.losses import BendingEnergy as BendingEnergyGPU
from losses.losses import Diffusion as DiffusionGPU
from losses.losses import Grad as GradGPU
from losses.losses_cpu import BendingEnergy as BendingEnergyCPU
from losses.losses_cpu import Diffusion as DiffusionCPU
from losses.losses_cpu import Grad as GradCPU


def _masked_channel_mean(values, mask):
    return (values * mask).sum() / (mask.sum() * values.shape[1])


class MaskAwareRegularizationNormalizationTests(unittest.TestCase):
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

    def _expected_grad(self, penalty, mask):
        dy = self.flow[:, :, 1:, :] - self.flow[:, :, :-1, :]
        dx = self.flow[:, :, :, 1:] - self.flow[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        if penalty == "l2":
            dy_values = dy.pow(2)
            dx_values = dx.pow(2)
        else:
            dy_values = dy.abs()
            dx_values = dx.abs()

        return (_masked_channel_mean(dy_values, mask_dy)
                + _masked_channel_mean(dx_values, mask_dx)) / 2.0

    def _expected_diffusion(self, mask):
        d2y = self.flow[:, :, 2:, :] - 2 * self.flow[:, :, 1:-1, :] + self.flow[:, :, :-2, :]
        d2x = self.flow[:, :, :, 2:] - 2 * self.flow[:, :, :, 1:-1] + self.flow[:, :, :, :-2]
        dxy = (self.flow[:, :, 1:, 1:] - self.flow[:, :, 1:, :-1]
               - self.flow[:, :, :-1, 1:] + self.flow[:, :, :-1, :-1])

        m_d2y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
        m_d2x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
        m_dxy = mask[:, :, 1:, 1:] * mask[:, :, 1:, :-1] * mask[:, :, :-1, 1:] * mask[:, :, :-1, :-1]

        loss = (_masked_channel_mean(d2y.pow(2), m_d2y)
                + _masked_channel_mean(d2x.pow(2), m_d2x)
                + 2 * _masked_channel_mean(dxy.pow(2), m_dxy))
        return loss / 4.0

    def _expected_bending_energy(self, mask):
        d2y = self.flow[:, :, 2:, :] - 2 * self.flow[:, :, 1:-1, :] + self.flow[:, :, :-2, :]
        d2x = self.flow[:, :, :, 2:] - 2 * self.flow[:, :, :, 1:-1] + self.flow[:, :, :, :-2]
        dxy = (self.flow[:, :, 1:, 1:] - self.flow[:, :, 1:, :-1]
               - self.flow[:, :, :-1, 1:] + self.flow[:, :, :-1, :-1])

        m_d2y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
        m_d2x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
        m_dxy = mask[:, :, 1:, 1:] * mask[:, :, 1:, :-1] * mask[:, :, :-1, 1:] * mask[:, :, :-1, :-1]

        return (_masked_channel_mean(d2y.pow(2), m_d2y)
                + _masked_channel_mean(d2x.pow(2), m_d2x)
                + 2 * _masked_channel_mean(dxy.pow(2), m_dxy))

    def test_full_mask_matches_unmasked_grad(self):
        for grad_cls in (GradGPU, GradCPU):
            for penalty in ("l1", "l2"):
                with self.subTest(implementation=grad_cls.__module__, penalty=penalty):
                    reg = grad_cls(penalty=penalty)
                    masked = reg(self.flow, mask=self.full_mask)
                    unmasked = reg(self.flow)
                    self.assertTrue(torch.isclose(masked, unmasked, atol=1e-6))

    def test_partial_mask_matches_expected_grad(self):
        for grad_cls in (GradGPU, GradCPU):
            for penalty in ("l1", "l2"):
                with self.subTest(implementation=grad_cls.__module__, penalty=penalty):
                    reg = grad_cls(penalty=penalty)
                    masked = reg(self.flow, mask=self.partial_mask)
                    expected = self._expected_grad(penalty, self.partial_mask)
                    self.assertTrue(torch.isclose(masked, expected, atol=1e-6))

    def test_full_mask_matches_unmasked_diffusion(self):
        for reg_cls in (DiffusionGPU, DiffusionCPU):
            with self.subTest(implementation=reg_cls.__module__):
                reg = reg_cls()
                masked = reg(self.flow, mask=self.full_mask)
                unmasked = reg(self.flow)
                self.assertTrue(torch.isclose(masked, unmasked, atol=1e-6))

    def test_partial_mask_matches_expected_diffusion(self):
        for reg_cls in (DiffusionGPU, DiffusionCPU):
            with self.subTest(implementation=reg_cls.__module__):
                reg = reg_cls()
                masked = reg(self.flow, mask=self.partial_mask)
                expected = self._expected_diffusion(self.partial_mask)
                self.assertTrue(torch.isclose(masked, expected, atol=1e-6))

    def test_full_mask_matches_unmasked_bending_energy(self):
        for reg_cls in (BendingEnergyGPU, BendingEnergyCPU):
            with self.subTest(implementation=reg_cls.__module__):
                reg = reg_cls()
                masked = reg(self.flow, mask=self.full_mask)
                unmasked = reg(self.flow)
                self.assertTrue(torch.isclose(masked, unmasked, atol=1e-6))

    def test_partial_mask_matches_expected_bending_energy(self):
        for reg_cls in (BendingEnergyGPU, BendingEnergyCPU):
            with self.subTest(implementation=reg_cls.__module__):
                reg = reg_cls()
                masked = reg(self.flow, mask=self.partial_mask)
                expected = self._expected_bending_energy(self.partial_mask)
                self.assertTrue(torch.isclose(masked, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
