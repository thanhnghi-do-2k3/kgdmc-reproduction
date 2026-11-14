import torch

from kgdmc.models import DMR, MTC


def test_dmr_forward_backward_finite() -> None:
    torch.manual_seed(42)
    model = DMR(input_dim=8, latent_dim=4)
    result = model(torch.randn(3, 8), torch.randn(3, 8))
    result.loss.backward()
    assert result.image_score.shape == (3,)
    assert torch.isfinite(result.loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mtc_all_granularities_and_ranking() -> None:
    model = MTC(dim=4, relation_dim=4)
    out = model(torch.randn(2, 4), torch.tensor([1, 2]), torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))
    assert set(out) == {"local", "global_structure", "global_path", "total"}
    loss = model.ranking_loss(out["total"], out["total"].flip(0))
    loss.backward()
    assert torch.isfinite(loss)

