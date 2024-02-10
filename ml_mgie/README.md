# ML-MGIE Packaging

**Work In Progress**: refacto, package, simplify dependencies, make compatible with MPS, CUDA and CPU
- by Paul Asquin

## Installation 
```bash
make venv
source venv_*/bin/activate
poetry install 
```

## Models download
Temporary from unofficial [huggingface.co/paulasquin/ml-mgie](https://huggingface.co/paulasquin/ml-mgie) for simplification convenience.
```bash
git lfs install
git clone https://huggingface.co/paulasquin/ml-mgie ./data
```

## Demo
```bash
poetry run python ml_mgie/demo/inference.py
```

## Typing check and tests
```bash
poetry run make tests
```

## Usage
```bash
poetry run python -m ml_mgie.main --input_path _input/0.jpg --instruction "make the frame red" --output_path red_glasses.jpg --max_size 512
```

```bash
poetry run python -m ml_mgie.app
```