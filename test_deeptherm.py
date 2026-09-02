import numpy as np
import pandas as pd
import pytest
import torch

from deeptherm.data import ThermochemistryDataset, collate_graphs, split_indices
from deeptherm.features import ATOM_FDIM, BOND_FDIM, descriptor_size, smiles_to_graph
from deeptherm.models import DeepThermModel, ModelConfig
from deeptherm.training import (
    TrainingConfig,
    masked_mse,
    predict_ensemble,
    regression_metrics,
    train_ensemble,
)


def make_batch(smiles=("CCO", "[CH3]", "[He]"), descriptor="none"):
    targets = np.asarray([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    dataset = ThermochemistryDataset(smiles, targets, descriptor)
    return collate_graphs([dataset[index] for index in range(len(dataset))])


def test_graph_has_directed_reverse_bonds_and_radical_features():
    graph = smiles_to_graph("[CH3]CO")
    assert graph.atom_features.shape[1] == ATOM_FDIM
    assert graph.bond_features.shape[1] == BOND_FDIM
    assert graph.bond_src.shape[0] == 2 * 2
    for index, reverse in enumerate(graph.bond_reverse):
        assert graph.bond_reverse[reverse] == index
        assert graph.bond_src[index] == graph.bond_dst[reverse]


def test_forward_attention_is_registered_and_handles_bondless_molecule():
    batch = make_batch()
    config = ModelConfig(
        atom_dim=ATOM_FDIM,
        bond_dim=BOND_FDIM,
        descriptor_dim=0,
        num_tasks=2,
        hidden_size=32,
        num_attention_heads=4,
    )
    model = DeepThermModel(config)
    attention_parameter_ids = {id(parameter) for parameter in model.global_attention.parameters()}
    all_parameter_ids = {id(parameter) for parameter in model.parameters()}
    assert attention_parameter_ids <= all_parameter_ids
    output = model(batch)
    assert output.shape == (3, 2)
    loss = masked_mse(output, batch.targets, batch.target_mask)
    loss.backward()
    assert model.global_attention.attention.in_proj_weight.grad is not None


def test_fixed_descriptors_are_concatenated():
    batch = make_batch(descriptor="morgan")
    model = DeepThermModel(
        ModelConfig(
            atom_dim=ATOM_FDIM,
            bond_dim=BOND_FDIM,
            descriptor_dim=descriptor_size("morgan"),
            num_tasks=2,
            hidden_size=32,
            num_attention_heads=4,
        )
    )
    assert model(batch).shape == (3, 2)


def test_transfer_loads_encoder_but_not_task_head():
    source = DeepThermModel(
        ModelConfig(ATOM_FDIM, BOND_FDIM, 0, 14, hidden_size=32, num_attention_heads=4)
    )
    target = DeepThermModel(
        ModelConfig(ATOM_FDIM, BOND_FDIM, 1024, 9, hidden_size=32, num_attention_heads=4)
    )
    target_head_before = target.ffn[-1].weight.detach().clone()
    target.load_pretrained_encoder(source.state_dict())
    assert torch.equal(target.encoder.atom_embedding.weight, source.encoder.atom_embedding.weight)
    assert torch.equal(
        target.global_attention.attention.in_proj_weight,
        source.global_attention.attention.in_proj_weight,
    )
    assert torch.equal(target.ffn[-1].weight, target_head_before)


def test_complexity_and_functional_group_splits_do_not_leak():
    smiles = ["C", "CC", "CCC", "CCCC", "CCO", "CCOO", "COO", "OOC"]
    train, valid, test = split_indices(smiles, "complexity", (0.5, 0.25, 0.25), 0)
    heavy = lambda i: smiles_to_graph(smiles[i]).atom_features.shape[0]
    assert max(map(heavy, train)) <= min(map(heavy, test))
    train, valid, test = split_indices(smiles, "functional-group", (0.5, 0.25, 0.25), 0, "OO")
    assert set(train).isdisjoint(test) and set(valid).isdisjoint(test)
    assert all("OO" not in smiles[index] for index in np.concatenate([train, valid]))
    assert all("OO" in smiles[index] for index in test)


def test_masked_metrics_ignore_missing_targets():
    actual = np.asarray([[1.0, np.nan], [3.0, 2.0]])
    predicted = np.asarray([[2.0, 99.0], [2.0, 2.0]])
    metrics = regression_metrics(actual, predicted)
    assert metrics["per_task"][0]["mae"] == pytest.approx(1.0)
    assert metrics["per_task"][1]["mae"] == pytest.approx(0.0)


def test_pretrain_transfer_finetune_and_weighted_ensemble(tmp_path):
    smiles = ["C", "CC", "CCC", "CCCC", "CO", "CCO", "C=C", "C#C", "[CH3]", "O"]
    qm9_tasks = [f"qm9_{index:02d}" for index in range(14)]
    pretrain_frame = pd.DataFrame({"smiles": smiles})
    for index, task in enumerate(qm9_tasks):
        pretrain_frame[task] = np.arange(len(smiles), dtype=np.float32) + index
    model_options = {
        "hidden_size": 16,
        "depth": 2,
        "num_attention_heads": 4,
        "ffn_hidden_size": 16,
        "ffn_num_layers": 2,
        "dropout": 0.0,
    }
    training = TrainingConfig(epochs=1, batch_size=4, patience=1, device="cpu")
    pretrain_results = train_ensemble(
        pretrain_frame,
        "smiles",
        qm9_tasks,
        tmp_path / "pretrain",
        "none",
        model_options,
        training,
        split_fractions=(0.6, 0.2, 0.2),
    )

    finetune_frame = pd.DataFrame(
        {
            "smiles": smiles,
            "Hf_298": np.linspace(-10, 10, len(smiles)),
            "S_298": np.linspace(20, 30, len(smiles)),
        }
    )
    fine_results = train_ensemble(
        finetune_frame,
        "smiles",
        ["Hf_298", "S_298"],
        tmp_path / "finetune",
        "morgan",
        model_options,
        training,
        split_fractions=(0.6, 0.2, 0.2),
        ensemble_size=2,
        pretrained_checkpoint=pretrain_results[0]["checkpoint"],
    )
    predictions, task_names, weights = predict_ensemble(
        [result["checkpoint"] for result in fine_results], smiles[:2], device_name="cpu"
    )
    assert predictions.shape == (2, 2)
    assert task_names == ["Hf_298", "S_298"]
    assert np.isfinite(predictions).all()
    assert weights.sum() == pytest.approx(1.0)
