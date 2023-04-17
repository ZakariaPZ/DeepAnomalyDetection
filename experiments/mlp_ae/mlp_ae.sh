#!/bin/bash
#SBATCH --partition=csc401
#SBATCH --gres=gpu
#SBATCH --time=5:00:00

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae1.yml > exp_results/dense_ae_1.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae2.yml > exp_results/dense_ae_2.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae3.yml > exp_results/dense_ae_3.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae4.yml > exp_results/dense_ae_4.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae5.yml > exp_results/dense_ae_5.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae6.yml > exp_results/dense_ae_6.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae7.yml > exp_results/dense_ae_7.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae8.yml > exp_results/dense_ae_8.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae9.yml > exp_results/dense_ae_9.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae10.yml > exp_results/dense_ae_10.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae11.yml > exp_results/dense_ae_11.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_ae/mlp_ae12.yml > exp_results/dense_ae_12.txt 2>&1