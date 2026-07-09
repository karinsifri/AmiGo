# AmiGo

> [!WARNING]
> **Branching meshes are not supported yet.** Shapes with limbs, ears, or other appendages
> (i.e. anything that isn't a single blob-like surface) will not produce a correct pattern.
> This feature is still in development — stay tuned!

Python implementation of the paper
[AmiGo: Computational Design of Amigurumi Crochet Patterns](https://mirelabc.github.io/publications/AmiGo_lores.pdf)
by Michal Edelstein, Hila Peleg, Shachar Itzhaky, and Mirela Ben-Chen (SCF '22).

AmiGo is an algorithm for converting a 3d triangle mesh into human-readable amigurumi crochet pattern.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate      # on Windows
source .venv/bin/activate   # on macOS/Linux
pip install -r requirements.txt
```

The mesh file picker (used when no mesh path is given) relies on `tkinter`, which ships with
most Python installations. On some Linux distributions it needs to be installed separately
(e.g. `apt install python3-tk`).

## Usage

Run the full pipeline from the project root:

```bash
python main.py
```

This will:
1. Open a file picker to select a mesh (unless a mesh path is given via config/CLI).
2. Open an interactive 3D viewer to pick a seed vertex (unless a seed is configured).
3. Compute the row/column parameterization, the crochet graph, and the stitch instructions,
   displaying each intermediate result in its own window. Each plot stops the execution, close the plot window to resume the execution.
4. Display the final, folded crochet instructions as a text figure.

### Configuration

Pipeline parameters are defined in and can be set in three ways, in increasing priority:

1. `config.yaml` at the project root (loaded by default).
2. An alternate YAML file passed via `--config path/to/file.yaml`.
3. Individual CLI flags:

```bash
python main.py --mesh_path meshes/zzmushroom_josh_r.obj --seed 82 --stitch_size 0.04 --use_creases true
```

Run `python main.py --help` for the full list of options, or `--print_config` to see the
fully resolved configuration without running the pipeline.

## Citation
If you find this code useful, please cite our paper:


```bibtex
@inproceedings{10.1145/3559400.3562005,
author = {Edelstein, Michal and Peleg, Hila and Itzhaky, Shachar and Ben-Chen, Mirela},
title = {AmiGo: Computational Design of Amigurumi Crochet Patterns},
year = {2022},
isbn = {9781450398725},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3559400.3562005},
doi = {10.1145/3559400.3562005},
booktitle = {Proceedings of the 7th Annual ACM Symposium on Computational Fabrication},
articleno = {5},
numpages = {11},
keywords = {geometry processing, crochet, computational knitting},
location = {Seattle, WA, USA},
series = {SCF '22}
}
```

```bibtex
@inproceedings{Edelstein_2022, series={SCF ’22},
   title={AmiGo: Computational Design of Amigurumi Crochet Patterns},
   url={http://dx.doi.org/10.1145/3559400.3562005},
   DOI={10.1145/3559400.3562005},
   booktitle={Proceedings of the 7th Annual ACM Symposium on Computational Fabrication},
   publisher={ACM},
   author={Edelstein, Michal and Peleg, Hila and Itzhaky, Shachar and Ben-Chen, Mirela},
   year={2022},
   month=Oct, pages={1–11},
   collection={SCF ’22} }
```

## License

Licensed under [CC BY-NC-SA 4.0](LICENSE).
