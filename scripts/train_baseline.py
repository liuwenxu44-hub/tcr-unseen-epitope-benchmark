from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.data import (  # noqa: E402
    UNK_HLA,
    DatasetSpec,
    build_training_data_bundle,
    resolve_partition_paths,
)
from src.dynamic_negative_audit import (  # noqa: E402
    validate_dynamic_inputs,
    validate_dynamic_partition_rows,
)
from src.model_cnn import TwoTowerCnn  # noqa: E402
from src.metrics import query_level_classification_metrics  # noqa: E402
from src.train import (  # noqa: E402
    choose_device,
    evaluate_model,
    fit_with_early_stopping,
    _file_sha256,
    load_model_from_checkpoint,
    set_global_seed,
    validate_distance_aware_five_fold_n1_inputs,
    validate_degree_matched_five_fold_inputs,
    validate_distance_aware_n1_inputs,
    validate_mixed_negative_manifest,
    validate_negative_sampling_manifest,
    validate_n2_integrity,
    validate_partition_interaction_isolation,
    validate_query_balanced_five_fold_n1_inputs,
    validate_unseen_epitope_isolation,
    write_metrics,
    write_per_query_metrics,
    write_predictions,
)


def _model_config(config: dict[str, Any], hla_vocab_size: int) -> dict[str, Any]:
    model = config["model"]
    return {
        "hla_vocab_size": hla_vocab_size,
        "aa_vocab_size": int(model["aa_vocab_size"]),
        "aa_embedding_dim": int(model["aa_embedding_dim"]),
        "cnn_channels": int(model["cnn_channels"]),
        "hla_embedding_dim": int(model["hla_embedding_dim"]),
        "kernel_sizes": list(model["kernel_sizes"]),
        "mlp_hidden_dims": list(model["mlp_hidden_dims"]),
        "padding_idx": int(model["padding_idx"]),
        "dropout": float(model["dropout"]),
    }


def _validate_reference_training_config(
    config: dict[str, Any], project_dir: Path
) -> dict[str, Any] | None:
    source = config.get("validation", {}).get("reference_training_config")
    if source is None:
        return None
    reference_path = project_dir / source["path"]
    observed_sha = _file_sha256(reference_path)
    if observed_sha != source["sha256"]:
        raise ValueError("reference training config SHA-256 mismatch")
    with reference_path.open("r", encoding="utf-8") as handle:
        reference = yaml.safe_load(handle)
    sections = (
        "seed",
        "sequence",
        "dataloader",
        "model",
        "optimizer",
        "training",
        "runtime",
    )
    mismatches = [
        section for section in sections if config[section] != reference[section]
    ]
    if mismatches:
        raise ValueError(
            "training configuration differs from reference: " + ", ".join(mismatches)
        )
    return {
        "path": source["path"],
        "sha256": observed_sha,
        "matching_sections": list(sections),
        "all_sections_match": True,
    }


def run_training(
    config_path: Path,
    *,
    project_dir: Path = PROJECT_DIR,
    device_override: str | None = None,
) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    reference_training_validation = _validate_reference_training_config(
        config, project_dir
    )
    seed = int(config["seed"])
    set_global_seed(seed)
    requested_device = device_override or config["runtime"]["device"]
    device = choose_device(requested_device)
    experiment = config["experiment"]
    spec = DatasetSpec(
        tier=experiment["tier"],
        positive_split_strategy=experiment["positive_split_strategy"],
        negative_strategy=experiment["negative_strategy"],
    )
    dynamic_input_validation = None
    if spec.negative_strategy == "DYNAMIC":
        dynamic_input_validation = validate_dynamic_inputs(config, project_dir)
        print(
            json.dumps(
                {"dynamic_input_validation": dynamic_input_validation},
                sort_keys=True,
            ),
            flush=True,
        )
    bundle = build_training_data_bundle(config, project_dir, spec)
    rows_by_partition = {
        partition: dataset.rows for partition, dataset in bundle.datasets.items()
    }
    data_validation = None
    if spec.positive_split_strategy == "unseen_epitope":
        data_validation = validate_unseen_epitope_isolation(
            {
                partition: dataset.rows
                for partition, dataset in bundle.datasets.items()
            }
        )
        print(
            json.dumps(
                {"unseen_epitope_validation": data_validation}, sort_keys=True
            ),
            flush=True,
        )
    if spec.positive_split_strategy == "distance_aware":
        validation_config = config["validation"]
        interaction_validation = validate_partition_interaction_isolation(
            rows_by_partition
        )
        fold_id = validation_config.get("fold_id")
        if fold_id is None:
            distance_validation = validate_distance_aware_n1_inputs(
                positive_manifest_path=project_dir
                / validation_config["positive_splits_manifest"],
                negative_manifest_path=project_dir
                / validation_config["negative_sampling_manifest"],
                positive_directory=project_dir
                / config["data"]["positive_split_directory"],
                negative_directory=project_dir
                / config["data"]["negative_directory"],
                positive_registry_path=project_dir
                / validation_config["positive_registry"],
                partition_rows=rows_by_partition,
                data_spec=spec,
                seed=seed,
                maximum_similarity=float(
                    validation_config[
                        "maximum_cross_partition_peptide_similarity"
                    ]
                ),
            )
            distance_validation_key = "distance_aware_inputs"
        else:
            if experiment.get("fold_id") != fold_id:
                raise ValueError("experiment and validation fold IDs disagree")
            partition_paths = {
                partition: resolve_partition_paths(
                    config, project_dir, spec, partition
                )
                for partition in ("train", "validation", "test")
            }
            if "degree_matched_manifest" in validation_config:
                distance_validation = validate_degree_matched_five_fold_inputs(
                    project_dir=project_dir,
                    fold_id=str(fold_id),
                    manifest_path=project_dir
                    / validation_config["degree_matched_manifest"],
                    expected_manifest_sha256=validation_config[
                        "degree_matched_manifest_sha256"
                    ],
                    positive_registry_path=project_dir
                    / validation_config["positive_registry"],
                    partition_paths=partition_paths,
                    partition_rows=rows_by_partition,
                    data_spec=spec,
                )
                distance_validation_key = "degree_matched_5fold_inputs"
            elif "query_balanced_manifest" in validation_config:
                distance_validation = validate_query_balanced_five_fold_n1_inputs(
                    project_dir=project_dir,
                    fold_id=str(fold_id),
                    manifest_path=project_dir
                    / validation_config["query_balanced_manifest"],
                    positive_registry_path=project_dir
                    / validation_config["positive_registry"],
                    partition_paths=partition_paths,
                    partition_rows=rows_by_partition,
                    data_spec=spec,
                    seed=seed,
                    per_query_cap=int(validation_config["per_query_cap"]),
                    maximum_similarity=float(
                        validation_config[
                            "maximum_cross_partition_peptide_similarity"
                        ]
                    ),
                )
                distance_validation_key = "query_balanced_5fold_inputs"
            else:
                distance_validation = validate_distance_aware_five_fold_n1_inputs(
                    fold_id=str(fold_id),
                    positive_manifest_path=project_dir
                    / validation_config["positive_splits_manifest"],
                    negative_manifest_path=project_dir
                    / validation_config["negative_sampling_manifest"],
                    positive_directory=project_dir
                    / config["data"]["positive_split_directory"],
                    negative_directory=project_dir
                    / config["data"]["negative_directory"],
                    positive_registry_path=project_dir
                    / validation_config["positive_registry"],
                    partition_paths=partition_paths,
                    partition_rows=rows_by_partition,
                    data_spec=spec,
                    seed=seed,
                    maximum_similarity=float(
                        validation_config[
                            "maximum_cross_partition_peptide_similarity"
                        ]
                    ),
                )
                distance_validation_key = "distance_aware_5fold_inputs"
        expected_train_hlas = {
            row["hla"].strip().upper() for row in rows_by_partition["train"]
        }
        fitted_hlas = set(bundle.hla_vocab.hla_to_id) - {UNK_HLA}
        if fitted_hlas != expected_train_hlas:
            raise ValueError("HLA vocabulary was not fitted exclusively from train")
        hla_validation = {
            "fitted_partition": "train",
            "known_hla_count": len(fitted_hlas),
            "validation_unknown_hla_count": sum(
                row["hla"].strip().upper() not in fitted_hlas
                for row in rows_by_partition["validation"]
            ),
            "test_unknown_hla_count": sum(
                row["hla"].strip().upper() not in fitted_hlas
                for row in rows_by_partition["test"]
            ),
        }
        data_validation = {
            distance_validation_key: distance_validation,
            "interaction_isolation": interaction_validation,
            "hla_mapping": hla_validation,
        }
        if reference_training_validation is not None:
            data_validation[
                "reference_training_config"
            ] = reference_training_validation
        print(
            json.dumps({"distance_aware_validation": data_validation}, sort_keys=True),
            flush=True,
        )
    if spec.negative_strategy == "DYNAMIC":
        interaction_validation = validate_partition_interaction_isolation(
            rows_by_partition
        )
        if data_validation is None:
            data_validation = {}
        data_validation["dynamic_inputs"] = dynamic_input_validation
        data_validation["interaction_isolation"] = interaction_validation
        dynamic_integrity = validate_dynamic_partition_rows(
            rows_by_partition,
            project_dir / config["data"]["positive_registry"],
        )
        data_validation["dynamic_integrity"] = dynamic_integrity
        print(
            json.dumps(
                {
                    "dynamic_interaction_isolation": interaction_validation,
                    "dynamic_integrity": dynamic_integrity,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    validation_config = config.get("validation", {})
    if (
        "negative_sampling_manifest" in validation_config
        and spec.positive_split_strategy != "distance_aware"
    ):
        interaction_validation = validate_partition_interaction_isolation(
            rows_by_partition
        )
        negative_directory = project_dir / config["data"]["negative_directory"]
        manifest_validation = validate_negative_sampling_manifest(
            manifest_path=project_dir
            / validation_config["negative_sampling_manifest"],
            negative_directory=negative_directory,
            partition_rows=rows_by_partition,
            data_spec=spec,
            seed=seed,
        )
        if data_validation is None:
            data_validation = {}
        data_validation["interaction_isolation"] = interaction_validation
        data_validation["negative_manifest"] = manifest_validation
        print(
            json.dumps(
                {
                    "interaction_isolation": interaction_validation,
                    "negative_manifest": manifest_validation,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if "mixed_negative_manifest" in validation_config:
        interaction_validation = validate_partition_interaction_isolation(
            rows_by_partition
        )
        mixed_validation = validate_mixed_negative_manifest(
            manifest_path=project_dir
            / validation_config["mixed_negative_manifest"],
            positive_directory=project_dir
            / config["data"]["positive_split_directory"],
            negative_directory=project_dir
            / config["data"]["negative_directory"],
            positive_registry_path=project_dir
            / validation_config["positive_registry"],
            partition_rows=rows_by_partition,
            data_spec=spec,
            seed=seed,
        )
        if data_validation is None:
            data_validation = {}
        data_validation["interaction_isolation"] = interaction_validation
        data_validation["mixed_manifest"] = mixed_validation
        print(
            json.dumps(
                {
                    "interaction_isolation": interaction_validation,
                    "mixed_manifest": mixed_validation,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if spec.negative_strategy == "N2":
        registry_path = project_dir / config["validation"]["positive_registry"]
        n2_validation = validate_n2_integrity(
            rows_by_partition,
            registry_path,
        )
        if data_validation is None:
            data_validation = {}
        data_validation["n2_integrity"] = n2_validation
        print(
            json.dumps({"n2_validation": n2_validation}, sort_keys=True),
            flush=True,
        )
    model_config = _model_config(config, len(bundle.hla_vocab))
    model = TwoTowerCnn(**model_config).to(device)
    output_dir = project_dir / config["output"]["run_dir"]
    training = config["training"]
    result = fit_with_early_stopping(
        model=model,
        train_loader=bundle.dataloaders["train"],
        validation_loader=bundle.dataloaders["validation"],
        device=device,
        output_dir=output_dir,
        model_config=model_config,
        amino_acid_vocab=bundle.amino_acid_vocab,
        hla_vocab=bundle.hla_vocab,
        data_spec=spec,
        seed=seed,
        learning_rate=float(config["optimizer"]["learning_rate"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["patience"]),
        early_stopping_metric=str(training["early_stopping_metric"]),
    )

    best_model, checkpoint = load_model_from_checkpoint(
        result.checkpoint_path, device
    )
    early_stopping_metric = str(training["early_stopping_metric"])
    per_query_rows = None
    if early_stopping_metric == "validation_query_macro_auprc":
        best_validation_result = evaluate_model(
            best_model, bundle.dataloaders["validation"], device
        )
        validation_macro, validation_per_query = query_level_classification_metrics(
            best_validation_result.predictions
        )
        if (
            abs(best_validation_result.auroc - float(result.best_validation["auroc"]))
            > 1e-12
            or abs(
                best_validation_result.auprc
                - float(result.best_validation["auprc"])
            )
            > 1e-12
            or abs(
                float(validation_macro["auprc"])
                - float(result.best_validation["query_macro"]["auprc"])
            )
            > 1e-12
        ):
            raise ValueError("best-checkpoint validation metrics are not reproducible")

        # Test is first touched only after the best checkpoint is fixed.
        test_result = evaluate_model(best_model, bundle.dataloaders["test"], device)
        test_macro, test_per_query = query_level_classification_metrics(
            test_result.predictions
        )
        validation_metrics: dict[str, Any] = {
            "pooled": best_validation_result.metrics_dict(),
            "query_macro": validation_macro,
        }
        test_metrics: dict[str, Any] = {
            "pooled": test_result.metrics_dict(),
            "query_macro": test_macro,
        }
        per_query_rows = [
            {"partition": "validation", **row} for row in validation_per_query
        ] + [{"partition": "test", **row} for row in test_per_query]
    else:
        # Test is intentionally first touched after model selection.
        test_result = evaluate_model(best_model, bundle.dataloaders["test"], device)
        validation_metrics = result.best_validation
        test_metrics = test_result.metrics_dict()
    metrics = {
        "schema_version": 1,
        "seed": seed,
        "device": str(device),
        "data_spec": checkpoint["data_spec"],
        "best_epoch": result.best_epoch,
        "early_stopping_metric": early_stopping_metric,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    if result.dynamic_sampling_audit_path is not None:
        metrics["dynamic_sampling_audit"] = {
            "file": result.dynamic_sampling_audit_path.name,
            "epochs": len(result.history),
        }
    if data_validation is not None:
        metrics["data_validation"] = data_validation
    write_metrics(output_dir / "metrics.json", metrics)
    write_predictions(output_dir / "test_predictions.csv", test_result.predictions)
    if per_query_rows is not None:
        write_per_query_metrics(output_dir / "per_query_metrics.csv", per_query_rows)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a configured two-tower CNN baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "baseline_cnn.yaml",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args()
    run_training(arguments.config, device_override=arguments.device)


if __name__ == "__main__":
    main()
