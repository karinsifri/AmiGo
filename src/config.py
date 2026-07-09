"""Configuration handling for the AmiGo pipeline: YAML/CLI parsing and mesh file picking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jsonargparse import ActionConfigFile, ArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    """Configurable parameters for the AmiGo pipeline.

    Args:
        mesh_path: Path to the mesh file to process; if not given, a file picker dialog opens to select one.
        seed: Seed vertex index; if negative, pick interactively in the 3D viewer.
        stitch_size: Spacing between sampled stitch points.
        use_creases: Classify stitches as BLO/FLO using curvature.
    """
    mesh_path: Optional[str] = None
    seed: int = -1
    stitch_size: float = 0.04
    use_creases: bool = False


def pick_mesh_file() -> str:
    """Open a native file picker dialog for selecting a mesh file."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a mesh file",
        initialdir=str(PROJECT_ROOT / "meshes"),
        filetypes=[("Mesh files", "*.obj *.off *.stl *.ply"), ("All files", "*.*")],
    )
    root.destroy()
    return file_path


def parse_config() -> PipelineConfig:
    """Parse pipeline configuration from config.yaml, an optional --config override, and CLI flags.

    Falls back to a file picker dialog for mesh_path when it isn't set by any of the above.
    """
    parser = ArgumentParser(description="Configurable parameters for the AmiGo pipeline.",
                            default_config_files=[str(PROJECT_ROOT / "config.yaml")])
    parser.add_argument("--config", action=ActionConfigFile, help="Path to a YAML config file")
    parser.add_class_arguments(PipelineConfig, nested_key=None, default=PipelineConfig())
    cfg = parser.parse_args()

    mesh_path = cfg.mesh_path or pick_mesh_file()
    if not mesh_path:
        raise RuntimeError("No mesh file was selected")

    mesh_path = Path(mesh_path)
    if not mesh_path.is_absolute():
        mesh_path = PROJECT_ROOT / mesh_path

    return PipelineConfig(mesh_path=str(mesh_path), seed=cfg.seed, stitch_size=cfg.stitch_size,
                          use_creases=cfg.use_creases)
