from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import trimesh as tm
from jsonargparse import ActionConfigFile, ArgumentParser

from plots import (
    plot_mesh_picker,
    plot_row_column_order,
    plot_crochet_graph,
    plot_flat_mesh,
    plot_stitch_rows,
    plot_final_instructions,
)
from src.order_functions import compute_row_column_order
from src.crochet_graph import get_crochet_graph
from src.instructions.loop_folding import Program
from src.instructions.create_instructions import dtw_to_stitches, create_final_instructions

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PipelineConfig:
    """Configurable parameters for the AmiGo pipeline.

    Args:
        mesh_path: Path to the mesh file to process.
        seed: Seed vertex index; if negative, pick interactively in the 3D viewer.
        stitch_size: Spacing between sampled stitch points.
        use_creases: Classify stitches as BLO/FLO using curvature.
    """
    mesh_path: str = "meshes/zzmushroom_josh_r.obj"
    seed: int = -1
    stitch_size: float = 0.04
    use_creases: bool = True


def parse_config() -> PipelineConfig:
    parser = ArgumentParser(description=__doc__, default_config_files=[str(PROJECT_ROOT / "config.yaml")])
    parser.add_argument("--config", action=ActionConfigFile, help="Path to a YAML config file")
    parser.add_class_arguments(PipelineConfig, nested_key=None, default=PipelineConfig())
    cfg = parser.parse_args()

    mesh_path = Path(cfg.mesh_path)
    if not mesh_path.is_absolute():
        mesh_path = PROJECT_ROOT / mesh_path
    cfg.mesh_path = str(mesh_path)

    return PipelineConfig(mesh_path=cfg.mesh_path, seed=cfg.seed, stitch_size=cfg.stitch_size,
                          use_creases=cfg.use_creases)


def run_pipeline(config: PipelineConfig) -> str:
    mesh = tm.load_mesh(config.mesh_path)

    seed = config.seed
    if seed < 0:
        plotter, picked_vert = plot_mesh_picker(mesh)
        plotter.show()
        if not picked_vert:
            raise RuntimeError("No seed vertex was picked")
        seed = picked_vert[-1]

    cut_mesh, row_order, column_order, path = compute_row_column_order(mesh, seed)

    row_plotter, col_plotter = plot_row_column_order(cut_mesh, row_order, column_order, path)
    row_plotter.show()
    col_plotter.show()

    plot_flat_mesh(cut_mesh, row_order, column_order, config.stitch_size).show()

    cg = get_crochet_graph(cut_mesh, row_order, column_order, config.stitch_size, config.use_creases)
    plot_crochet_graph(cg).show()

    row_stitch_sequences = [dtw_to_stitches(dtw, bflo)[::(-1) ** i]
                            for i, (dtw, bflo) in enumerate(zip(cg.connectivity, cg.split_creases))]
    plot_stitch_rows(cg.connectivity, [" ".join(seq) for seq in row_stitch_sequences], cg.split_creases)

    folded_stitch_rows = [f"{Program.fold(stitch_seq)} ({row_conn[-1][1] + 1})"
                          for stitch_seq, row_conn in zip(row_stitch_sequences, cg.connectivity)]
    plot_stitch_rows(cg.connectivity, folded_stitch_rows, cg.split_creases)

    instructions_text = create_final_instructions(folded_stitch_rows)
    plot_final_instructions(instructions_text)
    plt.show()

    return instructions_text


def main() -> None:
    config = parse_config()
    run_pipeline(config)


if __name__ == "__main__":
    main()
