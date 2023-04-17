#!/bin/bash
#SBATCH --partition=csc401
#SBATCH --gres=gpu
#SBATCH --time=5:00:00

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_1.yml > exp_results/conv_ae_1.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_2.yml > exp_results/conv_ae_2.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_3.yml > exp_results/conv_ae_3.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_4.yml > exp_results/conv_ae_4.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_5.yml > exp_results/conv_ae_5.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_6.yml > exp_results/conv_ae_6.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_7.yml > exp_results/conv_ae_7.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_8.yml > exp_results/conv_ae_8.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_9.yml > exp_results/conv_ae_9.txt 2>&1

python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_10.yml > exp_results/conv_ae_10.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_11.yml > exp_results/conv_ae_11.txt 2>&1
python3 main.py fit --trainer experiments/trainer.yml --data experiments/data.yml --model experiments/conv_ae/ae_conv_exp_12.yml > exp_results/conv_ae_12.txt 2>&1