"""Properties LoRA fine-tuning relies on. Each test fails if the matching bug comes back."""

import importlib.util
import os

import torch
import torch.nn as nn

# Load lora.py on its own: the training package __init__ pulls in the full trainer and its dependencies.
_path = os.path.join(os.path.dirname(__file__), "..", "src", "training", "lora.py")
_spec = importlib.util.spec_from_file_location("lora", _path)
lora = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lora)
LoRALayer, LoRALinear = lora.LoRALayer, lora.LoRALinear
inject_lora_into_model, merge_lora_weights = lora.inject_lora_into_model, lora.merge_lora_weights


class Attn(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, 2 * d)  # out != in, catches transposed merge maths
        self.norm = nn.LayerNorm(2 * d)

    def forward(self, x):
        return self.norm(self.to_v(x)) + self.to_q(x).sum() + self.to_k(x).sum()


def _trained_model():
    torch.manual_seed(0)
    model = inject_lora_into_model(Attn(), rank=4, target_modules=["to_q", "to_k", "to_v"])
    for m in model.modules():  # stand-in for training: give B non-zero values
        if isinstance(m, LoRALayer):
            nn.init.normal_(m.lora_B)
    return model


def test_fresh_adapter_leaves_output_unchanged():
    torch.manual_seed(0)
    base = Attn()
    x = torch.randn(3, 16)
    expected = base(x)
    model = inject_lora_into_model(base, rank=4, target_modules=["to_q", "to_k", "to_v"])
    assert torch.allclose(model(x), expected, atol=1e-6)


def test_only_adapters_train():
    model = _trained_model()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_A", "lora_B")) for n in trainable), trainable


def test_merge_keeps_output_identical_and_unmerge_restores():
    model = _trained_model().eval()
    x = torch.randn(3, 16)
    before = model(x)
    merge_lora_weights(model, merge=True)
    assert torch.allclose(model(x), before, atol=1e-5)
    merge_lora_weights(model, merge=False)
    assert torch.allclose(model(x), before, atol=1e-5)


def test_lora_linear_applies_adapter_in_eval_mode():
    torch.manual_seed(0)
    layer = LoRALinear(16, 8, rank=4)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(3, 16)
    layer.train()
    train_out = layer(x)
    layer.eval()
    assert torch.allclose(layer(x), train_out, atol=1e-6)
    assert not torch.allclose(layer(x), layer.linear(x))


def test_lora_linear_freezes_base():
    layer = LoRALinear(16, 8, rank=4)
    assert not any(p.requires_grad for p in layer.linear.parameters())


def test_adapter_matches_base_dtype():
    layer = LoRALayer(nn.Linear(16, 8).to(torch.float64), rank=4)
    assert layer.lora_A.dtype == layer.lora_B.dtype == torch.float64
    layer(torch.randn(2, 16, dtype=torch.float64))
