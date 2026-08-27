"""Offline CPU tests for DCCNet's paper-facing model components.

Run with:
    python -m unittest discover -s tests -p test_model_components.py -v
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from depth_anything.dpt import DepthAnything
from models.models import AIFE, DCCNet, DGModel_final, GPD
from utils.attention_module import AGSA, DeformableSpatialAttention, ModifiedChannelAttention


class _ZeroDeformConv(nn.Module):
    """Offline deterministic replacement for a DeformConv2d forward call."""

    def forward(self, x, offset):
        del offset
        return x[:, :1].new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])


class _TinyVGGFeatures(nn.Sequential):
    """43-layer VGG-compatible feature sequence with inexpensive 1x1 convs."""

    def __init__(self):
        layers = [nn.Conv2d(3, 256, 1), nn.ReLU(inplace=True), nn.MaxPool2d(4)]
        layers.extend(nn.Identity() for _ in range(20))  # first encoder: 23 layers
        layers.extend((nn.Conv2d(256, 512, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2)))
        layers.extend(nn.Identity() for _ in range(7))  # second encoder: 10 layers
        layers.extend((nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2)))
        layers.extend(nn.Identity() for _ in range(7))  # third encoder: 10 layers
        assert len(layers) == 43
        super().__init__(*layers)


class _DummyDepthPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x.mean(dim=1) * self.scale


class _FakeDINOBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, 3 * dim)


class _FakeDINO(nn.Module):
    """Small local backbone used only to test local checkpoint loading."""

    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeDINOBlock(dim)])

    def get_intermediate_layers(self, x, n, return_class_token=True):
        batch_size, _, height, width = x.shape
        tokens = x.new_zeros(batch_size, (height // 14) * (width // 14), self.blocks[0].attn.qkv.in_features)
        if return_class_token:
            cls = x.new_zeros(batch_size, tokens.shape[-1])
            return tuple((tokens, cls) for _ in range(n))
        return tuple(tokens for _ in range(n))


class TestModelComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_aife_reconstructs_features_and_preserves_legacy_logits_shape(self):
        module = AIFE(mem_size=5, mem_dim=4)
        features = torch.randn(2, 4, 3, 5, requires_grad=True)
        reconstruction, logits = module(features, torch.ones_like(features))
        self.assertEqual(reconstruction.shape, features.shape)
        self.assertEqual(logits.shape, (2, 5, 15))
        reconstruction.square().mean().backward()
        self.assertIsNotNone(module.mem.grad)
        self.assertIsNotNone(module.query_proj.weight.grad)

        # Older GPD checkpoints contain the prototype key only.  The AIFE
        # compatibility loader supplies identity Eq. (10) projections.
        legacy_state = {
            key: value for key, value in module.state_dict().items() if not key.startswith(("query_proj.", "key_proj.", "value_proj."))
        }
        restored = AIFE(mem_size=5, mem_dim=4)
        restored.load_state_dict(legacy_state, strict=True)
        self.assertTrue(torch.equal(restored.mem, module.mem.detach()))

    def test_agsa_uses_identity_preserving_equations_five_and_eight(self):
        x = torch.ones(1, 4, 3, 3)
        channel = ModifiedChannelAttention(4, ratio=2, use_learnable_avg_pool=False)
        for parameter in channel.fc.parameters():
            nn.init.zeros_(parameter)
        # W_c = sigmoid(0) = 0.5, therefore Eq. (5) gives 1.5 * x.
        self.assertTrue(torch.allclose(channel(x), x * 1.5))

        spatial = DeformableSpatialAttention(kernel_size=3)
        spatial.deform_conv = _ZeroDeformConv()
        # W_s = sigmoid(0) = 0.5, therefore Eq. (8) gives 1.5 * x.
        self.assertTrue(torch.allclose(spatial(x), x * 1.5))

        agsa = AGSA(4, ratio=2, spatial_kernel_size=3, use_learnable_avg_pool=False)
        agsa.ca = channel
        agsa.sa = spatial
        self.assertTrue(torch.allclose(agsa(x), x * 2.25))

    def test_dccnet_aliases_forward_and_frozen_normalized_depth_prior(self):
        prior = _DummyDepthPrior()
        model = DCCNet(
            pretrained=False,
            mem_size=5,
            mem_dim=8,
            deterministic=False,
            depth_anything=prior,
            vgg_features=_TinyVGGFeatures(),
        )
        self.assertIs(GPD, DCCNet)
        self.assertIs(DGModel_final, DCCNet)
        self.assertIs(model.aife, model.memory)
        self.assertIs(model.agsa, model.cbam_attention)
        state_keys = set(model.state_dict())
        self.assertIn("memory.mem", state_keys)
        self.assertTrue(any(key.startswith("cbam_attention.") for key in state_keys))

        model.train()
        self.assertFalse(prior.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in prior.parameters()))
        # AGSA is independently tested above; an identity here keeps this
        # end-to-end CPU test independent of torchvision's compiled op.
        model.cbam_attention = nn.Identity()
        model.eval()
        with torch.no_grad():
            density, aux = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(density.shape, (1, 1, 64, 64))
        self.assertEqual(aux.shape, (1, 1, 4, 4))

        expected = (torch.full((1, 3, 1, 1), 0.5) - model._depth_mean) / model._depth_std
        self.assertIsNotNone(prior.last_input)
        self.assertEqual(prior.last_input.shape[-2:], (70, 70))
        self.assertTrue(torch.allclose(prior.last_input[:, :, :1, :1], expected))

    def test_depthanything_loads_explicit_local_checkpoint_without_hub(self):
        config = {"encoder": "vits", "features": 8, "out_channels": [8, 8, 8, 8]}
        source = DepthAnything(config=config, backbone=_FakeDINO(dim=8))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as stream:
            checkpoint_path = stream.name
        try:
            torch.save(source.state_dict(), checkpoint_path)
            restored = DepthAnything.from_pretrained(
                checkpoint_path=checkpoint_path,
                config=config,
                backbone=_FakeDINO(dim=8),
            )
            self.assertEqual(set(source.state_dict()), set(restored.state_dict()))
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
