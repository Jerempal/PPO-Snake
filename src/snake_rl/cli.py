import argparse

from snake_rl.config import ExperimentConfig, with_environment_overrides


def add_map_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--map-size", type=int, help="Côté de la carte carrée jouable.")
    parser.add_argument("--grid-size", type=int, help="Côté fixe du canvas d'observation.")
    parser.add_argument("--width", type=int, help="Largeur de la carte jouable.")
    parser.add_argument("--height", type=int, help="Hauteur de la carte jouable.")


def apply_map_arguments(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    return with_environment_overrides(
        config,
        map_size=args.map_size,
        grid_size=args.grid_size,
        width=getattr(args, "width", None),
        height=getattr(args, "height", None),
    )
