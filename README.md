# DeepTherm
This is the repository for the paper "DeepTherm: A Unified Deep Learning Approach to ‎Thermochemistry Prediction for Gas-phase Molecules‎".

DeepTherm is a deep learning framework designed to predict thermochemical properties, such as enthalpy of formation, entropy, and heat capacity for diverse molecular species. The project integrates directed message-passing networks and global attention mechanisms to capture both local and long-range dependencies in molecular graphs.

All the Supplementary data of the paper are provided in the Supplementary data.zip.

The code was built based on DMPNN. Thanks a lot for their code sharing!


## Dependencies

```bash
conda env create -f environment.yml
conda activate deeptherm
pip install -e .
pytest
```


## Data

The first column of the CSV does not need to be fixed, but there must be a SMILES column (default name: `smiles`). The remaining specified columns represent floating-point targets; null values ​​or `NaN`s are masked during
multi-task loss calculation.

```csv
smiles,target_1,target_2
CCO,-56.2,67.4
[CH3],35.0,
```

## QM9 Pretraining


```bash
python train.py pretrain \
  --data data/qm9.csv \
  --targets target01 target02 target03 target04 target05 target06 target07 \
            target08 target09 target10 target11 target12 target13 target14 \
  --save-dir runs/qm9 \
  --epochs 100 --batch-size 128 \
  --learning-rate 1e-4 --weight-decay 1e-5 \
  --ensemble-size 1
```


## Training

```bash
python train.py finetune \
  --data data/thermochemistry.csv \
  --targets Hf_298 S_298 Cp_300 Cp_400 Cp_500 Cp_600 Cp_800 Cp_1000 Cp_1500 \
  --pretrained runs/qm9/model_00.pt \
  --save-dir runs/deeptherm \
  --descriptor morgan \
  --ensemble-size 10
```

By default, `finetune` generates 10 sub-models. The main text of the paper lists only the eight types of descriptors tested and does not specify the final selected configuration for each target; furthermore, the repository does not include Supporting Information. Therefore, the ECFP 1024-bit format explicitly specified in the main text is used by default. If CDS, OCHEM, or DScribe features (as used in the paper) are already available, the `fixed_descriptors` can be extended after pre-calculation in CSV format without requiring changes to the model's concatenation interface.

Example of hierarchical evaluation:：

```bash
python train.py finetune --data data/thermo.csv --save-dir runs/complexity \
  --pretrained runs/qm9/model_00.pt --split complexity --ensemble-size 1

python train.py finetune --data data/thermo.csv --save-dir runs/peroxide \
  --pretrained runs/qm9/model_00.pt --split functional-group --test-smarts "OO" \
  --ensemble-size 1
```

## Weighted ensemble predictions

```bash
python predict.py \
  --data data/predict.csv \
  --checkpoint-dir runs/deeptherm \
  --output predictions.csv
```

The program automatically loads `model_*.pt` files and calculates weights using the validation set MAE values:
`w_i = (1 / MAE_i) / sum_j(1 / MAE_j)`. It then outputs a `pred_<target_name>` column.


## Acknowledgement
This code was developed at the King Abdullah University of Science and Technology (KAUST).
