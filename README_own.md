# Installation

For a clean conda environment with python 3.9 or 3.10.

```sh
pip3 install torch torchvision torchaudio
conda install mpi4py
pip3 install -e .
```

# Tips

To avoid conflicts with packages named in the same way as dirs in root, use

```python
import importlib
hf_datasets = importlib.import_module('datasets')
```