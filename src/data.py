from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .vocab import AminoAcidVocab
from .tcrue.mixed_negatives import (
    MixedCandidate,
    candidate_selection_sha256,
    select_mixed_candidates,
)


UNK_HLA = "<UNK_HLA>"
UNK_HLA_ID = 0
PARTITIONS = ("train", "validation", "test")
TIERS = ("development", "strict")
POSITIVE_SPLIT_STRATEGIES = ("random", "unseen_epitope", "distance_aware")
NEGATIVE_STRATEGIES = ("N1", "N2", "MIXED", "DYNAMIC", "DEGREE_MATCHED")
REQUIRED_COLUMNS = {"tcr_cdr3_beta", "epitope", "hla", "label"}


@dataclass(frozen=True)
class DatasetSpec:
    tier: str
    positive_split_strategy: str
    negative_strategy: str

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"Unknown tier: {self.tier}")
        if self.positive_split_strategy not in POSITIVE_SPLIT_STRATEGIES:
            raise ValueError(
                f"Unknown positive split strategy: {self.positive_split_strategy}"
            )
        if self.negative_strategy not in NEGATIVE_STRATEGIES:
            raise ValueError(f"Unknown negative strategy: {self.negative_strategy}")


@dataclass(frozen=True)
class PartitionPaths:
    positive_csv: Path
    negative_csv: Path


@dataclass(frozen=True)
class TrainingDataBundle:
    amino_acid_vocab: AminoAcidVocab
    hla_vocab: "HlaVocab"
    datasets: dict[str, "TcrEpitopeDataset"]
    dataloaders: dict[str, DataLoader]


class HlaVocab:
    """Stable HLA mapping fitted from training rows only."""

    def __init__(self, hla_to_id: Mapping[str, int]) -> None:
        mapping = dict(hla_to_id)
        if mapping.get(UNK_HLA) != UNK_HLA_ID:
            raise ValueError("HLA mapping must reserve ID 0 for UNK_HLA")
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("HLA IDs must be contiguous")
        self.hla_to_id = mapping
        self.id_to_hla = {index: hla for hla, index in mapping.items()}

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, str]]) -> "HlaVocab":
        observed = sorted({row["hla"].strip().upper() for row in rows if row["hla"].strip()})
        mapping = {UNK_HLA: UNK_HLA_ID}
        mapping.update({hla: index for index, hla in enumerate(observed, start=1)})
        return cls(mapping)

    def __len__(self) -> int:
        return len(self.hla_to_id)

    def encode(self, hla: str) -> int:
        return self.hla_to_id.get(hla.strip().upper(), UNK_HLA_ID)

    def decode(self, hla_id: int) -> str:
        return self.id_to_hla.get(int(hla_id), UNK_HLA)


def load_training_data_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("training data config must contain a YAML mapping")
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("training data config schema_version must be 1")
    sequence = config.get("sequence", {})
    if int(sequence.get("tcr_max_length", 0)) <= 0:
        raise ValueError("tcr_max_length must be positive")
    if int(sequence.get("epitope_max_length", 0)) <= 0:
        raise ValueError("epitope_max_length must be positive")
    loader = config.get("dataloader", {})
    if int(loader.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive")
    if int(loader.get("num_workers", -1)) < 0:
        raise ValueError("num_workers must be non-negative")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")


def _resolve_project_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def resolve_partition_paths(
    config: Mapping[str, Any],
    project_dir: Path,
    spec: DatasetSpec,
    partition: str,
) -> PartitionPaths:
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    data_config = config["data"]
    positive_dir = _resolve_project_path(
        project_dir, data_config["positive_split_directory"]
    )
    negative_dir = _resolve_project_path(
        project_dir, data_config["negative_directory"]
    )
    partition_files = data_config.get("partition_files")
    if partition_files is not None:
        if not isinstance(partition_files, Mapping):
            raise ValueError("data.partition_files must be a mapping")
        entry = partition_files.get(partition)
        if not isinstance(entry, Mapping):
            raise ValueError(f"missing explicit files for partition: {partition}")
        positive_file = entry.get("positive")
        negative_file = entry.get("negative")
        if not isinstance(positive_file, str) or not isinstance(negative_file, str):
            raise ValueError(
                f"explicit partition files must be strings for: {partition}"
            )
        return PartitionPaths(
            positive_csv=positive_dir / positive_file,
            negative_csv=negative_dir / negative_file,
        )
    prefix = f"{spec.tier}_{spec.positive_split_strategy}_{partition}"
    return PartitionPaths(
        positive_csv=positive_dir / f"{prefix}.csv",
        negative_csv=negative_dir / f"{prefix}_{spec.negative_strategy.lower()}.csv",
    )


def _read_csv(path: Path, expected_label: int, source: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input has no header: {path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"Invalid label at {path}:{row_number}") from exc
            if label != expected_label:
                raise ValueError(
                    f"Expected label {expected_label} at {path}:{row_number}, observed {label}"
                )
            normalized = dict(row)
            normalized["tcr_cdr3_beta"] = row["tcr_cdr3_beta"].strip().upper()
            normalized["epitope"] = row["epitope"].strip().upper()
            normalized["hla"] = row["hla"].strip().upper()
            normalized["_source"] = source
            rows.append(normalized)
    return rows


def load_partition_rows(paths: PartitionPaths) -> list[dict[str, str]]:
    positives = _read_csv(paths.positive_csv, expected_label=1, source="positive")
    negatives = _read_csv(
        paths.negative_csv, expected_label=0, source="putative_negative"
    )
    return positives + negatives


class TcrEpitopeDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        amino_acid_vocab: AminoAcidVocab,
        hla_vocab: HlaVocab,
        *,
        tcr_max_length: int,
        epitope_max_length: int,
        spec: DatasetSpec,
        partition: str,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.amino_acid_vocab = amino_acid_vocab
        self.hla_vocab = hla_vocab
        self.tcr_max_length = tcr_max_length
        self.epitope_max_length = epitope_max_length
        self.spec = spec
        self.partition = partition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tcr = self.amino_acid_vocab.encode(
            row["tcr_cdr3_beta"], self.tcr_max_length
        )
        epitope = self.amino_acid_vocab.encode(
            row["epitope"], self.epitope_max_length
        )
        source = row.get("_source", "unknown")
        sample_id = (
            row.get("record_id", "")
            if source == "positive"
            else row.get("negative_id", "")
        )
        return {
            "tcr_ids": torch.tensor(tcr.token_ids, dtype=torch.long),
            "epitope_ids": torch.tensor(epitope.token_ids, dtype=torch.long),
            "hla_id": torch.tensor(self.hla_vocab.encode(row["hla"]), dtype=torch.long),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "tcr_original_length": torch.tensor(
                tcr.original_length, dtype=torch.long
            ),
            "tcr_effective_length": torch.tensor(
                tcr.effective_length, dtype=torch.long
            ),
            "epitope_original_length": torch.tensor(
                epitope.original_length, dtype=torch.long
            ),
            "epitope_effective_length": torch.tensor(
                epitope.effective_length, dtype=torch.long
            ),
            "metadata": {
                "sample_id": sample_id,
                "source": source,
                "tier": self.spec.tier,
                "strategy": self.spec.positive_split_strategy,
                "split": self.partition,
                "positive_split_strategy": self.spec.positive_split_strategy,
                "negative_strategy": self.spec.negative_strategy,
                "partition": self.partition,
                "pmhc_id": row.get("pmhc_id", ""),
            },
        }


class DynamicNegativeDataset(TcrEpitopeDataset):
    """Fixed positives with deterministic epoch-specific train negatives."""

    def __init__(
        self,
        positive_rows: Sequence[Mapping[str, str]],
        n1_rows: Sequence[Mapping[str, str]],
        n2_rows: Sequence[Mapping[str, str]],
        positive_registry: set[tuple[str, str, str]],
        forbidden_peptides: set[str],
        forbidden_interactions: set[tuple[str, str, str]],
        amino_acid_vocab: AminoAcidVocab,
        hla_vocab: HlaVocab,
        *,
        tcr_max_length: int,
        epitope_max_length: int,
        spec: DatasetSpec,
        base_seed: int,
    ) -> None:
        if spec.negative_strategy != "DYNAMIC" or spec.positive_split_strategy != "unseen_epitope":
            raise ValueError("dynamic dataset requires unseen_epitope DYNAMIC spec")
        self.base_seed = int(base_seed)
        self.current_epoch = -1
        self.selection_statistics: dict[str, int] = {}
        self.selection_sha256 = ""
        self.selected_negative_keys: tuple[tuple[str, str, str], ...] = ()

        def normalize(row: Mapping[str, str], expected_label: str) -> dict[str, str]:
            normalized = dict(row)
            normalized["tcr_cdr3_beta"] = row["tcr_cdr3_beta"].strip().upper()
            normalized["epitope"] = row["epitope"].strip().upper()
            normalized["hla"] = row["hla"].strip().upper()
            if str(row.get("label")) != expected_label:
                raise ValueError(f"dynamic input label must be {expected_label}")
            return normalized

        self.fixed_positive_rows = [normalize(row, "1") for row in positive_rows]
        positive_keys = [self._key(row) for row in self.fixed_positive_rows]
        if len(positive_keys) != len(set(positive_keys)):
            raise ValueError("dynamic positives contain duplicate interactions")
        train_peptides = {row["epitope"] for row in self.fixed_positive_rows}
        n1_map = {
            self._key(normalized): normalized
            for normalized in (normalize(row, "0") for row in n1_rows)
        }
        n2_map = {
            self._key(normalized): normalized
            for normalized in (normalize(row, "0") for row in n2_rows)
        }
        if len(n1_map) != len(n1_rows) or len(n2_map) != len(n2_rows):
            raise ValueError("dynamic candidate pools contain duplicate interactions")
        candidates = []
        for key in sorted(set(n1_map) | set(n2_map)):
            if key in positive_registry:
                raise ValueError(f"dynamic candidate intersects positive registry: {key}")
            if key in forbidden_interactions:
                raise ValueError(f"dynamic candidate uses held-out interaction: {key}")
            if key[1] in forbidden_peptides or key[1] not in train_peptides:
                raise ValueError(f"dynamic candidate uses held-out peptide: {key[1]}")
            in_n1 = key in n1_map
            in_n2 = key in n2_map
            candidates.append(
                MixedCandidate(
                    key=key,
                    partition="train",
                    source="both" if in_n1 and in_n2 else "N1" if in_n1 else "N2",
                    n1_negative_id=n1_map[key].get("negative_id", "") if in_n1 else "",
                    n2_negative_id=n2_map[key].get("negative_id", "") if in_n2 else "",
                )
            )
        self.dynamic_candidates = tuple(candidates)
        super().__init__(
            self.fixed_positive_rows,
            amino_acid_vocab,
            hla_vocab,
            tcr_max_length=tcr_max_length,
            epitope_max_length=epitope_max_length,
            spec=spec,
            partition="train",
        )
        self.set_epoch(0)

    @staticmethod
    def _key(row: Mapping[str, str]) -> tuple[str, str, str]:
        return (row["tcr_cdr3_beta"], row["epitope"], row["hla"])

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("dynamic epoch must be non-negative")
        epoch = int(epoch)
        selected, statistics = select_mixed_candidates(
            self.dynamic_candidates,
            len(self.fixed_positive_rows),
            seed=self.base_seed + epoch,
            partition="train",
        )
        negative_rows = []
        for candidate in selected:
            tcr, epitope, hla = candidate.key
            payload = "\0".join((str(self.base_seed), str(epoch), *candidate.key)).encode(
                "utf-8"
            )
            negative_rows.append(
                {
                    "negative_id": f"dynamic-negative:{hashlib.sha256(payload).hexdigest()[:20]}",
                    "tcr_cdr3_beta": tcr,
                    "epitope": epitope,
                    "hla": hla,
                    "pmhc_id": f"{epitope}|{hla}",
                    "label": "0",
                    "source": candidate.source,
                    "sampling_seed": str(self.base_seed + epoch),
                    "sampling_epoch": str(epoch),
                    "source_n1_negative_id": candidate.n1_negative_id,
                    "source_n2_negative_id": candidate.n2_negative_id,
                    "_source": "putative_negative",
                }
            )
        positives = []
        for row in self.fixed_positive_rows:
            normalized = dict(row)
            normalized["_source"] = "positive"
            positives.append(normalized)
        self.rows = positives + negative_rows
        self.current_epoch = epoch
        self.selection_statistics = statistics
        self.selection_sha256 = candidate_selection_sha256(selected)
        self.selected_negative_keys = tuple(candidate.key for candidate in selected)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "base_seed": self.base_seed,
            "current_epoch": self.current_epoch,
            "next_epoch": self.current_epoch + 1,
            "selection_sha256": self.selection_sha256,
            "selection_statistics": dict(self.selection_statistics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or int(state.get("base_seed", -1)) != self.base_seed:
            raise ValueError("dynamic sampling state is incompatible")
        self.set_epoch(int(state["current_epoch"]))
        if self.selection_sha256 != state.get("selection_sha256"):
            raise ValueError("restored dynamic selection SHA-256 mismatch")


class EpochDataLoader(DataLoader):
    """DataLoader whose sampling and shuffle order are both epoch-addressable."""

    def __init__(
        self,
        dataset: DynamicNegativeDataset,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        base_seed: int,
        pin_memory: bool = False,
    ) -> None:
        self.base_seed = int(base_seed)
        generator = torch.Generator()
        generator.manual_seed(self.base_seed)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=generator,
        )

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        if self.generator is None:
            raise ValueError("epoch DataLoader generator is unavailable")
        self.generator.manual_seed(self.base_seed + int(epoch))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "base_seed": self.base_seed,
            "dataset": self.dataset.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or int(state.get("base_seed", -1)) != self.base_seed:
            raise ValueError("epoch DataLoader state is incompatible")
        self.dataset.load_state_dict(state["dataset"])
        if self.generator is None:
            raise ValueError("epoch DataLoader generator is unavailable")
        self.generator.manual_seed(self.base_seed + self.dataset.current_epoch)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def build_training_data_bundle(
    config: Mapping[str, Any], project_dir: Path, spec: DatasetSpec
) -> TrainingDataBundle:
    _validate_config(config)
    if spec.negative_strategy == "DYNAMIC":
        return _build_dynamic_training_data_bundle(config, project_dir, spec)
    partition_rows = {
        partition: load_partition_rows(
            resolve_partition_paths(config, project_dir, spec, partition)
        )
        for partition in PARTITIONS
    }
    amino_acid_vocab = AminoAcidVocab()
    hla_vocab = HlaVocab.fit(partition_rows["train"])
    sequence_config = config["sequence"]
    datasets = {
        partition: TcrEpitopeDataset(
            rows,
            amino_acid_vocab,
            hla_vocab,
            tcr_max_length=int(sequence_config["tcr_max_length"]),
            epitope_max_length=int(sequence_config["epitope_max_length"]),
            spec=spec,
            partition=partition,
        )
        for partition, rows in partition_rows.items()
    }
    loader_config = config["dataloader"]
    shuffle_config = loader_config["shuffle"]
    dataloaders = {
        partition: make_dataloader(
            dataset,
            batch_size=int(loader_config["batch_size"]),
            shuffle=bool(shuffle_config[partition]),
            num_workers=int(loader_config["num_workers"]),
            seed=int(config["seed"]),
            pin_memory=bool(loader_config.get("pin_memory", False)),
        )
        for partition, dataset in datasets.items()
    }
    return TrainingDataBundle(
        amino_acid_vocab=amino_acid_vocab,
        hla_vocab=hla_vocab,
        datasets=datasets,
        dataloaders=dataloaders,
    )


def _build_dynamic_training_data_bundle(
    config: Mapping[str, Any], project_dir: Path, spec: DatasetSpec
) -> TrainingDataBundle:
    data_config = config["data"]
    positive_dir = _resolve_project_path(
        project_dir, data_config["positive_split_directory"]
    )
    negative_dir = _resolve_project_path(
        project_dir, data_config["negative_directory"]
    )
    mixed_dir = _resolve_project_path(
        project_dir, data_config["mixed_negative_directory"]
    )
    registry_path = _resolve_project_path(
        project_dir, data_config["positive_registry"]
    )
    prefix = f"{spec.tier}_{spec.positive_split_strategy}"
    positives = {
        partition: _read_csv(
            positive_dir / f"{prefix}_{partition}.csv",
            expected_label=1,
            source="positive",
        )
        for partition in PARTITIONS
    }
    train_n1 = _read_csv(
        negative_dir / f"{prefix}_train_n1.csv",
        expected_label=0,
        source="putative_negative",
    )
    train_n2 = _read_csv(
        negative_dir / f"{prefix}_train_n2.csv",
        expected_label=0,
        source="putative_negative",
    )
    fixed_negatives = {
        partition: _read_csv(
            mixed_dir / f"{prefix}_{partition}_mixed.csv",
            expected_label=0,
            source="putative_negative",
        )
        for partition in ("validation", "test")
    }
    registry_rows = _read_csv(
        registry_path, expected_label=1, source="positive_registry"
    )
    positive_registry = {
        DynamicNegativeDataset._key(row) for row in registry_rows
    }
    forbidden_peptides = {
        row["epitope"]
        for partition in ("validation", "test")
        for row in (*positives[partition], *fixed_negatives[partition])
    }
    forbidden_interactions = {
        DynamicNegativeDataset._key(row)
        for partition in ("validation", "test")
        for row in (*positives[partition], *fixed_negatives[partition])
    }
    amino_acid_vocab = AminoAcidVocab()
    hla_vocab = HlaVocab.fit(positives["train"])
    sequence_config = config["sequence"]
    train_dataset = DynamicNegativeDataset(
        positives["train"],
        train_n1,
        train_n2,
        positive_registry,
        forbidden_peptides,
        forbidden_interactions,
        amino_acid_vocab,
        hla_vocab,
        tcr_max_length=int(sequence_config["tcr_max_length"]),
        epitope_max_length=int(sequence_config["epitope_max_length"]),
        spec=spec,
        base_seed=int(config["base_seed"]),
    )
    datasets: dict[str, TcrEpitopeDataset] = {
        "train": train_dataset,
        **{
            partition: TcrEpitopeDataset(
                positives[partition] + fixed_negatives[partition],
                amino_acid_vocab,
                hla_vocab,
                tcr_max_length=int(sequence_config["tcr_max_length"]),
                epitope_max_length=int(sequence_config["epitope_max_length"]),
                spec=spec,
                partition=partition,
            )
            for partition in ("validation", "test")
        },
    }
    loader_config = config["dataloader"]
    dataloaders: dict[str, DataLoader] = {
        "train": EpochDataLoader(
            train_dataset,
            batch_size=int(loader_config["batch_size"]),
            shuffle=bool(loader_config["shuffle"]["train"]),
            num_workers=int(loader_config["num_workers"]),
            base_seed=int(config["base_seed"]),
            pin_memory=bool(loader_config.get("pin_memory", False)),
        ),
        **{
            partition: make_dataloader(
                datasets[partition],
                batch_size=int(loader_config["batch_size"]),
                shuffle=bool(loader_config["shuffle"][partition]),
                num_workers=int(loader_config["num_workers"]),
                seed=int(config["seed"]),
                pin_memory=bool(loader_config.get("pin_memory", False)),
            )
            for partition in ("validation", "test")
        },
    }
    return TrainingDataBundle(
        amino_acid_vocab=amino_acid_vocab,
        hla_vocab=hla_vocab,
        datasets=datasets,
        dataloaders=dataloaders,
    )
