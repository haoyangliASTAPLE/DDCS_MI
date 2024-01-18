exp=1
seeds=(0)
attack='vanilla'
loaders='vanilla_10'

# train evaluator
cmd="-m torch.distributed.launch --nproc_per_node=4 --use_env \
train.py
--seed 0
--exp ${exp}
--arch inceptionv3
--resolution 299
"
python $cmd
sleep 30

# perform experiments
for seed in "${seeds[@]}"; do
    echo "Running iteration $seed"

    # train target
    cmd="-m torch.distributed.launch --nproc_per_node=4 --use_env \
    train.py
    --seed ${seed}
    --exp ${exp}
    "
    python $cmd
    sleep 30

    # vanilla transfer
    cmd="-m torch.distributed.launch --nproc_per_node=4 --use_env \
    transfer.py
    --seed ${seed}
    --exp ${exp}
    --attack ${attack}
    "
    python $cmd
    sleep 30

    # recover
    cmd="-m torch.distributed.launch --nproc_per_node=4 --use_env \
    recover.py
    --seed ${seed}
    --exp ${exp}
    --attack ${attack}
    "
    python $cmd
    sleep 30

    # evaluate
    cmd="evaluate.py
    --seed ${seed}
    --exp ${exp}
    --loaders ${loaders}
    "
    CUDA_VISIBLE_DEVICES=0 python $cmd
    sleep 30

done