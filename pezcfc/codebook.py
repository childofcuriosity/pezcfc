"""Token-codebook filtering and exact cosine nearest-neighbour projection."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CodebookFilter:
    """A normalized codebook plus indices into the original vocabulary."""

    normalized_embeddings: torch.Tensor
    original_indices: torch.Tensor
    threshold: float

    @classmethod
    def from_embedding_layer(
        cls, embedding_layer: torch.nn.Embedding, threshold: float
    ) -> "CodebookFilter":
        """Keep tokens whose embedding-coordinate std is at least ``threshold``."""
        if threshold < 0:
            raise ValueError("std threshold must be non-negative")

        with torch.no_grad():
            weights = embedding_layer.weight
            token_stds = weights.std(dim=-1)
            indices = torch.nonzero(token_stds >= threshold, as_tuple=True)[0]
            if indices.numel() == 0:
                maximum = token_stds.max().item()
                raise ValueError(
                    f"std threshold {threshold} removes every token; "
                    f"maximum token std is {maximum:.6f}"
                )
            normalized = F.normalize(weights[indices], p=2, dim=-1)

        return cls(normalized, indices, float(threshold))


def project_nearest(
    current_embeddings: torch.Tensor,
    embedding_layer: torch.nn.Embedding,
    codebook_filter: Optional[CodebookFilter] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project ``[batch, length, dim]`` embeddings to exact cosine neighbours.

    Returned token IDs always index the original vocabulary, including when a
    filtered codebook is used. This is the central change relative to PEZ.
    """
    if current_embeddings.ndim != 3:
        raise ValueError("current_embeddings must have shape [batch, length, dim]")

    batch_size, sequence_length, embedding_dim = current_embeddings.shape
    if embedding_dim != embedding_layer.weight.shape[1]:
        raise ValueError("embedding dimensions do not match")

    with torch.no_grad():
        queries = F.normalize(
            current_embeddings.reshape(-1, embedding_dim), p=2, dim=-1
        )
        if codebook_filter is None:
            search_matrix = F.normalize(embedding_layer.weight, p=2, dim=-1)
            index_map = None
        else:
            search_matrix = codebook_filter.normalized_embeddings
            index_map = codebook_filter.original_indices

        local_indices = (queries @ search_matrix.T).argmax(dim=-1)
        token_indices = (
            index_map[local_indices] if index_map is not None else local_indices
        )
        token_indices = token_indices.reshape(batch_size, sequence_length)
        projected = embedding_layer(token_indices)

    return projected, token_indices

