#!/bin/bash
#SBATCH --partition=csc401
#SBATCH --gres=gpu
#SBATCH --time=5:00:00

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae1.yml > exp_results/dense_ae_1.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae2.yml > exp_results/dense_ae_2.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae3.yml > exp_results/dense_ae_3.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae4.yml > exp_results/dense_ae_4.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae5.yml > exp_results/dense_ae_5.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae6.yml > exp_results/dense_ae_6.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae7.yml > exp_results/dense_ae_7.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae8.yml > exp_results/dense_ae_8.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae9.yml > exp_results/dense_ae_9.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae10.yml > exp_results/dense_ae_10.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae11.yml > exp_results/dense_ae_11.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/mlp_vae/mlp_vae12.yml > exp_results/dense_ae_12.txt 2>&1