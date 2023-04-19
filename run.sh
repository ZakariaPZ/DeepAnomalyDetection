# run all the vae experiments in the background
# get the experiment name from the first argument passed to the script
# e.g. bash run.sh mlp_vae
e=$1
echo "Starting $e experiments"
# get the accelerator from the second argument passed to the script (put it to cpu if not specified)
a=$2
if [ -z "$a" ]
then
    a="auto"
fi
echo "Using $a accelerator"

# get the trainer from the third argument passed to the script (put it to trainer if not specified)
t=$3
if [ -z "$t" ]
then
    t="trainer.yml"
fi


# run the for all the models in the folder experiments/$e
for i in experiments/$e/*.yml
do  
    # get the file name without the extension
    i=$(basename $i .yml)
    echo "Starting experiment $e $i"
    screen -S vae-$i -dm python main.py fit --model experiments/$e/$i.yml --trainer experiments/$t --data experiments/data.yml --trainer.accelerator=$a --trainer.logger.init_args.name=$e-$i;
done