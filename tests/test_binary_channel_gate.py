import unittest

import torch

from models.binary_channel_gate import (
    PROTOCOL_VERSION,
    BinaryChannelGate,
    build_causal_gate_features,
    validate_gate_checkpoint,
)


class BinaryChannelGateTests(unittest.TestCase):
    def test_causal_augmented_features_match_validated_definition(self):
        torch.manual_seed(7)
        drift = torch.randn(1, 3, 5)
        stats = torch.randn(3, 5)
        frozen = torch.randn(1, 4, 3)
        fixed = frozen + torch.randn(1, 4, 3) * 0.1

        actual = build_causal_gate_features(
            drift, stats, frozen, fixed, "causal_augmented"
        )
        frozen_c = frozen.squeeze(0).transpose(0, 1)
        correction_c = (fixed - frozen).squeeze(0).transpose(0, 1)
        frozen_rms = frozen_c.pow(2).mean(dim=-1).sqrt()
        correction_rms = correction_c.pow(2).mean(dim=-1).sqrt()
        output_features = torch.stack(
            [
                frozen_c.mean(dim=-1),
                frozen_c.abs().mean(dim=-1),
                frozen_c.std(dim=-1, unbiased=False),
                correction_c.mean(dim=-1),
                correction_c.abs().mean(dim=-1),
                correction_rms,
                correction_rms / frozen_rms.clamp(min=1e-6),
            ],
            dim=-1,
        )
        expected = torch.cat(
            [drift.squeeze(0), stats, output_features], dim=-1
        ).unsqueeze(0)
        self.assertTrue(torch.equal(actual, expected))

    def test_decision_is_strict_fixed_zero_threshold(self):
        gate = BinaryChannelGate(2, hidden=0)
        with torch.no_grad():
            gate.network.weight.copy_(torch.tensor([[1.0, 0.0]]))
            gate.network.bias.zero_()
        features = torch.tensor([[[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0]]])
        expected = torch.tensor([[False, False, True]])
        self.assertTrue(torch.equal(gate.decisions(features), expected))

    def test_formal_checkpoint_requires_embedded_safety_mode(self):
        valid = {
            "protocol_version": PROTOCOL_VERSION,
            "selected_mode": "learned_regressor",
            "decision_threshold": 0.0,
        }
        validate_gate_checkpoint(valid)
        with self.assertRaises(ValueError):
            validate_gate_checkpoint({**valid, "selected_mode": None})
        with self.assertRaises(ValueError):
            validate_gate_checkpoint({**valid, "decision_threshold": 0.1})


if __name__ == "__main__":
    unittest.main()
