#!/bin/bash
#SBATCH --partition=csc401
#SBATCH --gres=gpu

python3 ae_test.py --type conv --n_layers_encoder 2 --encoder_width 392  --n_layers_decoder 2 --decoder_width 392  --epochs 20 --lr 1e-2 --normal_class $1 --version 1 > exp_results/conv_loi_$1/conv_ae_1.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 2 --encoder_width 784  --n_layers_decoder 2 --decoder_width 784  --epochs 20 --lr 1e-2 --normal_class $1 --version 2 > exp_results/conv_loi_$1/conv_ae_2.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 2 --encoder_width 1568 --n_layers_decoder 2 --decoder_width 1568 --epochs 20 --lr 1e-2 --normal_class $1 --version 3 > exp_results/conv_loi_$1/conv_ae_3.txt 2>&1

python3 ae_test.py --type conv --n_layers_encoder 1 --encoder_width 784 --n_layers_decoder 1 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 4 > exp_results/conv_loi_$1/conv_ae_4.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 1 --encoder_width 784 --n_layers_decoder 2 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 5 > exp_results/conv_loi_$1/conv_ae_5.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 1 --encoder_width 784 --n_layers_decoder 4 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 6 > exp_results/conv_loi_$1/conv_ae_6.txt 2>&1

python3 ae_test.py --type conv --n_layers_encoder 2 --encoder_width 784 --n_layers_decoder 1 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 7 > exp_results/conv_loi_$1/conv_ae_7.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 2 --encoder_width 784 --n_layers_decoder 2 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 8 > exp_results/conv_loi_$1/conv_ae_8.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 2 --encoder_width 784 --n_layers_decoder 4 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 9 > exp_results/conv_loi_$1/conv_ae_9.txt 2>&1

python3 ae_test.py --type conv --n_layers_encoder 4 --encoder_width 784 --n_layers_decoder 1 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 10 > exp_results/conv_loi_$1/conv_ae_10.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 4 --encoder_width 784 --n_layers_decoder 2 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 11 > exp_results/conv_loi_$1/conv_ae_11.txt 2>&1
python3 ae_test.py --type conv --n_layers_encoder 4 --encoder_width 784 --n_layers_decoder 4 --decoder_width 784 --epochs 20 --lr 1e-2 --normal_class $1 --version 12 > exp_results/conv_loi_$1/conv_ae_12.txt 2>&1