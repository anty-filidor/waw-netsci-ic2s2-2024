# Presentation and examples shown at WAW 2024

## Configuration of the runtime

Commands for Python:

```bash
conda create --name waw-24 python=3.12 --y && conda activate waw-24
pip install network-diffusion==0.15.0 ipykernel==6.29.4
python -m ipykernel install --user --name waw-24
```

To compile presentation use pdfTeX 3.141592653-2.6-1.40.26 (TeX Live 2024)

## Structure of the repository

```bash
.
├── README.md
├── example_i.ipynb   # a basic example of the network_diffusion
├── example_ii.ipynb  # an advanced example of the network_diffusion
├── my_experiment     # outdir for example_i.ipynb
├── networks          # dataset used in example_ii.ipynb
├── poster            # LaTeX sources used to build a poster presented at NetSci
├── presentation      # LaTeX sources for the presentation (WAW-24 & IC2S2 24)
└── utils             # helper srcipts used in example_ii.ipynb
```
