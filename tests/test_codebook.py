import pytest
import torch

from pezcfc.codebook import CodebookFilter, project_nearest


def test_filter_keeps_original_vocabulary_indices():
    layer = torch.nn.Embedding(4, 3)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    filtered = CodebookFilter.from_embedding_layer(layer, threshold=0.5)
    query = torch.tensor([[[0.0, 0.9, 0.1]]])
    projected, token_ids = project_nearest(query, layer, filtered)

    assert filtered.original_indices.tolist() == [1, 2, 3]
    assert token_ids.tolist() == [[2]]
    assert torch.equal(projected, layer(token_ids))


def test_zero_threshold_matches_unfiltered_projection():
    torch.manual_seed(7)
    layer = torch.nn.Embedding(10, 5)
    query = torch.randn(2, 3, 5)
    filtered = CodebookFilter.from_embedding_layer(layer, threshold=0.0)

    _, baseline_ids = project_nearest(query, layer)
    _, filtered_ids = project_nearest(query, layer, filtered)

    assert torch.equal(baseline_ids, filtered_ids)


def test_rejects_threshold_that_removes_every_token():
    layer = torch.nn.Embedding(2, 3)
    with pytest.raises(ValueError, match="removes every token"):
        CodebookFilter.from_embedding_layer(layer, threshold=100.0)

