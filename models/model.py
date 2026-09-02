from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, values.size(-1)))
    if index.numel():
        output.index_add_(0, index, values)
    return output


@dataclass(frozen=True)
class ModelConfig:
    atom_dim: int
    bond_dim: int
    descriptor_dim: int
    num_tasks: int
    hidden_size: int = 300
    depth: int = 3
    num_attention_heads: int = 4
    ffn_hidden_size: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.ffn_num_layers < 1:
            raise ValueError("ffn_num_layers must be at least 1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphBatch:
    atom_features: Tensor
    bond_features: Tensor
    bond_src: Tensor
    bond_dst: Tensor
    bond_reverse: Tensor
    atom_scope: Sequence[tuple[int, int]]
    descriptors: Tensor | None = None
    targets: Tensor | None = None
    target_mask: Tensor | None = None

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            atom_features=self.atom_features.to(device),
            bond_features=self.bond_features.to(device),
            bond_src=self.bond_src.to(device),
            bond_dst=self.bond_dst.to(device),
            bond_reverse=self.bond_reverse.to(device),
            atom_scope=self.atom_scope,
            descriptors=None if self.descriptors is None else self.descriptors.to(device),
            targets=None if self.targets is None else self.targets.to(device),
            target_mask=None if self.target_mask is None else self.target_mask.to(device),
        )


class DirectedBondMessagePassing(nn.Module):
    """Chemprop-style D-MPNN which excludes the reverse bond message."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden_size: int, depth: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.atom_embedding = nn.Linear(atom_dim, hidden_size)
        self.bond_embedding = nn.Linear(atom_dim + bond_dim, hidden_size)
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(max(depth - 1, 0))
        )
        self.atom_output = nn.Linear(2 * hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: GraphBatch) -> Tensor:
        atom_input = self.activation(self.atom_embedding(batch.atom_features))
        if batch.bond_src.numel() == 0:
            incoming = atom_input.new_zeros(atom_input.shape)
        else:
            bond_input = self.activation(
                self.bond_embedding(
                    torch.cat([batch.atom_features[batch.bond_src], batch.bond_features], dim=-1)
                )
            )
            message = bond_input
            for layer in self.message_layers:
                incoming_to_atom = _scatter_sum(message, batch.bond_dst, atom_input.size(0))
                # For i->j, aggregate k->i and explicitly remove j->i.
                directed_message = incoming_to_atom[batch.bond_src] - message[batch.bond_reverse]
                message = self.dropout(self.activation(bond_input + layer(directed_message)))
            incoming = _scatter_sum(message, batch.bond_dst, atom_input.size(0))

        atom_hidden = self.activation(self.atom_output(torch.cat([atom_input, incoming], dim=-1)))
        return self.dropout(atom_hidden)


class GlobalAttentionReadout(nn.Module):
    """Masked multi-head self-attention and learned pooling over each molecule."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.pooling_score = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_hidden: Tensor, atom_scope: Sequence[tuple[int, int]]) -> Tensor:
        if not atom_scope:
            return atom_hidden.new_empty((0, atom_hidden.size(-1)))
        max_atoms = max(size for _, size in atom_scope)
        padded = atom_hidden.new_zeros((len(atom_scope), max_atoms, atom_hidden.size(-1)))
        padding_mask = torch.ones(
            (len(atom_scope), max_atoms), dtype=torch.bool, device=atom_hidden.device
        )
        for molecule_index, (start, size) in enumerate(atom_scope):
            padded[molecule_index, :size] = atom_hidden.narrow(0, start, size)
            padding_mask[molecule_index, :size] = False

        attended, _ = self.attention(
            padded, padded, padded, key_padding_mask=padding_mask, need_weights=False
        )
        attended = self.attention_norm(padded + self.dropout(attended))
        attended = self.output_norm(attended + self.dropout(self.feed_forward(attended)))

        scores = self.pooling_score(attended).squeeze(-1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).masked_fill(padding_mask, 0.0)
        return torch.sum(attended * weights.unsqueeze(-1), dim=1)


class DeepThermModel(nn.Module):
    """End-to-end multi-task regression model described in the DeepTherm paper."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = DirectedBondMessagePassing(
            config.atom_dim,
            config.bond_dim,
            config.hidden_size,
            config.depth,
            config.dropout,
        )
        self.global_attention = GlobalAttentionReadout(
            config.hidden_size, config.num_attention_heads, config.dropout
        )

        input_size = config.hidden_size + config.descriptor_dim
        layers: list[nn.Module] = []
        for layer_index in range(config.ffn_num_layers - 1):
            layers.extend(
                [
                    nn.Linear(input_size if layer_index == 0 else config.ffn_hidden_size,
                              config.ffn_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        final_input = input_size if config.ffn_num_layers == 1 else config.ffn_hidden_size
        layers.append(nn.Linear(final_input, config.num_tasks))
        self.ffn = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() == 1:
                nn.init.zeros_(parameter)
            else:
                nn.init.xavier_normal_(parameter)
        # LayerNorm scales must start at one.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)

    def encode(self, batch: GraphBatch) -> Tensor:
        atoms = self.encoder(batch)
        return self.global_attention(atoms, batch.atom_scope)

    def forward(self, batch: GraphBatch) -> Tensor:
        molecule_vectors = self.encode(batch)
        if self.config.descriptor_dim:
            if batch.descriptors is None:
                raise ValueError("This checkpoint requires fixed molecular descriptors")
            molecule_vectors = torch.cat([molecule_vectors, batch.descriptors], dim=-1)
        return self.ffn(molecule_vectors)

    def load_pretrained_encoder(self, checkpoint_state: dict[str, Tensor]) -> tuple[list[str], list[str]]:
        """Loads only transferable graph/attention weights, never the task-specific FFN."""
        prefixes = ("encoder.", "global_attention.")
        transferable = {key: value for key, value in checkpoint_state.items() if key.startswith(prefixes)}
        current = self.state_dict()
        incompatible = [
            key for key, value in transferable.items() if key not in current or current[key].shape != value.shape
        ]
        if incompatible:
            raise ValueError(f"Incompatible pretrained encoder tensors: {incompatible}")
        result = self.load_state_dict(transferable, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)


def build_model(config: ModelConfig | dict) -> DeepThermModel:
    return DeepThermModel(config if isinstance(config, ModelConfig) else ModelConfig(**config))
