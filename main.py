import matplotlib.pyplot as plt
import trimesh as tm

from plots import (
    plot_mesh_picker,
    plot_row_column_order,
    plot_crochet_graph,
    plot_flat_mesh,
    plot_stitch_rows,
    plot_final_instructions,
)
from preprocess.cmcf import smooth_craters
from preprocess.scale import set_surface_area_to_one
from src.config import PipelineConfig, parse_config
from src.crochet_graph import get_crochet_graph
from src.instructions.create_instructions import create_final_instructions, get_row_stitch_sequences, \
    get_folded_rows
from src.order_functions import compute_row_column_order


def run_pipeline(config: PipelineConfig) -> str:
    """Run the pipeline end to end, displaying each intermediate plot, and return the final instructions."""
    mesh = tm.load_mesh(config.mesh_path)

    mesh = set_surface_area_to_one(mesh)
    mesh = smooth_craters(mesh)

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

    row_stitch_sequences = get_row_stitch_sequences(cg.connectivity, cg.split_creases)
    plot_stitch_rows(cg.connectivity, [" ".join(seq) for seq in row_stitch_sequences], cg.split_creases)

    folded_stitch_rows = get_folded_rows(row_stitch_sequences, cg.connectivity)
    plot_stitch_rows(cg.connectivity, folded_stitch_rows, cg.split_creases)

    instructions_text = create_final_instructions(folded_stitch_rows)
    plot_final_instructions(instructions_text)
    plt.show()

    return instructions_text


def main() -> None:
    """Parse the pipeline configuration and run it."""
    config = parse_config()
    run_pipeline(config)


if __name__ == "__main__":
    main()
