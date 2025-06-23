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

For any experiment, the _label_ and _PASSION_split_ files must be available in the _data_ folder, as well as the PASSION data.
Access to the data must be requested via PASSION's webpage: https://passionderm.github.io/.


### Experiment: Differential Diagnosis and Detecting Impetigo (Table 2 and 3)
> python -m src.evaluate_experiments --config_path configs/default.yaml --exp1 --exp2

### Experiment: Generalization across Collection Centers and Age Groups (Sec. 5, Paragraph 2 and 3)
> python -m src.evaluate_experiments --config_path configs/default.yaml --exp3 --exp4


### Fairness Evaluation as of Bachelor Thesis "Demographic Biases in Dermatology AI"
The fairness evaluation pipeline introduced in that thesis has been added as the evaluator class.
The fairness evaluation is run directly for the fine-tuning process. It can be disabled by setting the _detailed_evaluation_ config to false.

The evaluator can also be started independently, if an experiment output file is available in _assets/evaluation_, after some adaptions in the `__main__`-block.
Start the run using
> cd src/utils/
> python -m evaluator

#### Stratified splitting experiments: exp5 - 7
> python -m src.evaluate_experiments --config_path configs/default.yaml --exp5 --exp6 --exp7

In the background, those experiments will create stratified splits. Records in low support groups (support=1) will be added to the training set.

This creation could also be triggered independently, including a distribution analysis, using
> cd src/utils/
> python -m stratified_split_generator

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
