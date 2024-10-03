# PASSION for Dermatology
This repository contains the code to reproduce all evaluations in the paper "PASSION for Dermatology: Bridging the Diversity Gap with Pigmented Skin Images from Sub-Saharan Africa".

## Usage
Run `make` for a list of possible targets.

## Installation
Run this command for installation
`make install`

## Reproducibility of the Paper
To reproduce our experiments, we list the detailed comments needed for replicating each experiment below.
Note that our experiments were run on a DGX Workstation 1.
If less computational power is available, this would require adaptations of the configuration file.

### Experiment: Differential Diagnosis and Detecting Impetigo (Table 2 and 3)
> python -m src.evaluate_experiments --config_path configs/default.yaml --exp1 --exp2

### Experiment: Generalization across Collection Centers and Age Groups (Sec. 5, Paragraph 2 and 3)
> python -m src.evaluate_experiments --config_path configs/default.yaml --exp3 --exp4

## Code and test conventions
- `black` for code style
- `isort` for import sorting
- docstring style: `sphinx`
- `pytest` for running tests

### Development installation and configurations
To set up your dev environment run:
```bash
pip install -r requirements.txt
# install pre-commit hooks
pre-commit install
```
