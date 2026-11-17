"""Train the no-added-noise SGSSC stages using external train/val directories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgssc_amc.models import SGSSCConfig
from sgssc_amc.training import TrainingSettings, train_classifier, train_pretrain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("pretrain", "classifier"):
        command = commands.add_parser(stage)
        command.add_argument("--train-data", type=Path, required=True)
        command.add_argument("--val-data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
        command.add_argument("--steps", type=int, default=20000 if stage == "pretrain" else 2000)
        command.add_argument("--lr", type=float, default=1e-4 if stage == "pretrain" else 1e-3)
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--validate-every", type=int, default=250)
        command.add_argument("--seed", type=int, default=20261001)
        command.add_argument("--threads", type=int, default=4)
        if stage == "pretrain":
            command.add_argument("--validation-observations", type=int, default=512)
            command.add_argument("--model-config", type=Path, help="Optional SGSSCConfig JSON; default is the published architecture")
        else:
            command.add_argument("--pretrained", type=Path, required=True,
                                 help="New pretrain best.pt or public inference checkpoint; readout is initialized afresh")
            command.add_argument("--label-fraction", default="0.005")
            command.add_argument("--training-pool-size", type=int, default=242000)
    args = parser.parse_args(argv)
    settings = TrainingSettings(args.steps, args.lr, args.batch_size, args.validate_every,
                                args.seed, args.threads, getattr(args, "validation_observations", 512))
    if args.stage == "pretrain":
        config = SGSSCConfig(**json.loads(args.model_config.read_text(encoding="utf-8"))) if args.model_config else SGSSCConfig()
        report = train_pretrain(args.train_data, args.val_data, args.output, config, settings)
    else:
        report = train_classifier(args.train_data, args.val_data, args.pretrained, args.output, settings,
                                  args.label_fraction, args.training_pool_size)
    print(f"Completed {args.stage}: selected validation step {report['selected_step']}; report: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
