from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Ponyland launchers for the multiview transformer workflow.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--data-yaml", required=True, help="Cluster-side YOLO dataset YAML.")
    parser.add_argument(
        "--manifest-output",
        default="outputs/multiview_transformer/manifests/m4_scene_manifest.csv",
        help="Where the launcher should write the scene manifest.",
    )
    parser.add_argument("--quality-csv", default="", help="Optional cluster-side quality CSV to merge into the manifest.")
    parser.add_argument("--quality-key", default="file_name", help="Join key for the optional quality CSV.")
    parser.add_argument("--quality-columns", default="", help="Optional comma-separated quality columns to merge.")
    parser.add_argument("--launcher", choices=["bash", "slurm"], default="slurm", help="Launcher type to write.")
    parser.add_argument("--stage", choices=["train", "rank", "pipeline"], default="pipeline", help="Workflow stage to emit.")
    parser.add_argument("--python-executable", default="python", help="Python executable to use in the generated launcher.")
    parser.add_argument("--workspace-root", default=str(Path.cwd()), help="Workspace root for the generated launcher.")
    parser.add_argument("--job-name", default="mv-transformer", help="Slurm job name when --launcher=slurm.")
    parser.add_argument("--pair-top-k", type=int, default=16, help="Top-K singles to use when generating pair candidates.")
    parser.add_argument("--triple-top-k", type=int, default=16, help="Top-K singles to use when generating triple candidates.")
    parser.add_argument("--slurm-partition", default="", help="Optional Slurm partition.")
    parser.add_argument("--slurm-account", default="", help="Optional Slurm account.")
    parser.add_argument("--slurm-time", default="12:00:00", help="Optional Slurm wall-time.")
    parser.add_argument("--slurm-mem", default="32G", help="Optional Slurm memory request.")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=8, help="Optional Slurm CPU request.")
    parser.add_argument("--slurm-gres", default="gpu:1", help="Optional Slurm GRES request.")
    parser.add_argument(
        "--output-script",
        default="outputs/multiview_transformer/launchers/launch_pipeline_slurm.sh",
        help="Path of the launcher script to write.",
    )
    return parser.parse_args()


def quote(value: str) -> str:
    return shlex.quote(value)


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def as_posix_string(value: str) -> str:
    return str(value).replace("\\", "/")


def build_commands(args: argparse.Namespace, experiment_output_dir: Path) -> list[str]:
    commands: list[str] = []
    manifest_output = as_posix_string(args.manifest_output)
    checkpoint_path = (experiment_output_dir / "checkpoints" / "best.pt").as_posix()
    eval_dir = (experiment_output_dir / "eval" / "test").as_posix()
    single_summary = f"{eval_dir}/combo_1_summary.csv"

    if args.stage in {"train", "pipeline"}:
        manifest_command = [
            quote(args.python_executable),
            "multiview_transformer/build_scene_manifest.py",
            "--data-yaml",
            quote(args.data_yaml),
            "--output-csv",
            quote(manifest_output),
        ]
        if args.quality_csv:
            manifest_command.extend(["--quality-csv", quote(args.quality_csv)])
            manifest_command.extend(["--quality-key", quote(args.quality_key)])
        if args.quality_columns:
            manifest_command.extend(["--quality-columns", quote(args.quality_columns)])
        commands.append(" ".join(manifest_command))
        commands.append(
            " ".join(
                [
                    quote(args.python_executable),
                    "multiview_transformer/train.py",
                    "--config",
                    quote(str(Path(args.config).as_posix())),
                    "--manifest-path",
                    quote(manifest_output),
                ]
            )
        )

    if args.stage in {"rank", "pipeline"}:
        commands.append(
            " ".join(
                [
                    quote(args.python_executable),
                    "multiview_transformer/evaluate_sets.py",
                    "--checkpoint",
                    quote(checkpoint_path),
                    "--combo-size",
                    "1",
                    "--split",
                    "test",
                    "--require-complete",
                ]
            )
        )
        commands.append(
            " ".join(
                [
                    quote(args.python_executable),
                    "multiview_transformer/evaluate_sets.py",
                    "--checkpoint",
                    quote(checkpoint_path),
                    "--combo-size",
                    "2",
                    "--split",
                    "test",
                    "--require-complete",
                    "--shortlist-from",
                    quote(single_summary),
                    "--top-k",
                    str(args.pair_top_k),
                ]
            )
        )
        commands.append(
            " ".join(
                [
                    quote(args.python_executable),
                    "multiview_transformer/evaluate_sets.py",
                    "--checkpoint",
                    quote(checkpoint_path),
                    "--combo-size",
                    "3",
                    "--split",
                    "test",
                    "--require-complete",
                    "--shortlist-from",
                    quote(single_summary),
                    "--top-k",
                    str(args.triple_top_k),
                ]
            )
        )
        commands.append(
            " ".join(
                [
                    quote(args.python_executable),
                    "multiview_transformer/rank_viewpoints.py",
                    "--eval-dir",
                    quote(eval_dir),
                ]
            )
        )

    return commands


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config).resolve())
    experiment_output_dir = Path(as_posix_string(config["experiment"]["output_dir"]))
    output_script = Path(args.output_script).resolve()
    output_script.parent.mkdir(parents=True, exist_ok=True)
    log_dir = (experiment_output_dir / "logs").as_posix()

    lines = ["#!/usr/bin/env bash"]
    if args.launcher == "slurm":
        lines.extend(
            [
                f"#SBATCH --job-name={args.job_name}",
                f"#SBATCH --output={log_dir}/%x_%j.out",
                f"#SBATCH --error={log_dir}/%x_%j.err",
                "#SBATCH --ntasks=1",
            ]
        )
        if args.slurm_partition:
            lines.append(f"#SBATCH --partition={args.slurm_partition}")
        if args.slurm_account:
            lines.append(f"#SBATCH --account={args.slurm_account}")
        if args.slurm_time:
            lines.append(f"#SBATCH --time={args.slurm_time}")
        if args.slurm_mem:
            lines.append(f"#SBATCH --mem={args.slurm_mem}")
        if args.slurm_cpus_per_task > 0:
            lines.append(f"#SBATCH --cpus-per-task={args.slurm_cpus_per_task}")
        if args.slurm_gres:
            lines.append(f"#SBATCH --gres={args.slurm_gres}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {quote(as_posix_string(args.workspace_root))}",
            "export PYTHONUNBUFFERED=1",
            "",
        ]
    )
    lines.extend(build_commands(args=args, experiment_output_dir=experiment_output_dir))
    lines.append("")
    output_script.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_script}")


if __name__ == "__main__":
    main()
