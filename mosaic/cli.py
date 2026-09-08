"""Public command-line entry points for the MOSAIC workflow.

Commands import their implementation only after selection. CASO and baseline
commands therefore do not require the optional PromoSim dependencies.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module

COMMANDS = {
    "demo": ("mosaic.demo", "run a synthetic CPU training and prediction example"),
    "promosim": ("mosaic.promosim.generate", "generate random movie–seed training records"),
    "content-match": ("mosaic.caso.content_cli", "encode static user–movie content match"),
    "train-flow": ("mosaic.caso.flow_training", "train seed flow from synthetic k=0–10 masks"),
    "train": ("mosaic.caso.train", "train CASO in two stages, starting from scratch"),
    "optimize": ("mosaic.caso.optimize", "generate exact-budget CASO seed policies"),
    "predict": ("mosaic.caso.inference", "predict total acceptance for supplied seeds"),
    "evaluate": ("mosaic.promosim.evaluate", "evaluate supplied seeds with PromoSim replays"),
    "baselines": ("baselines.run_baselines", "generate topology, IC, and LT baseline seeds"),
    "content-baseline": ("baselines.run_content_baseline", "generate ContentMatch-Degree seeds"),
}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="mosaic", description="MOSAIC seed optimization and behavioral evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    # Each implementation owns its parser, including command-specific help.
    selected = parser.parse_args(arguments[:1])
    module = import_module(COMMANDS[selected.command][0])
    module.main(arguments[1:])
