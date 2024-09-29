#!/bin/sh
env="StarCraft2"
map="2m_vs_1z"
algo="mdpo"
exp="test"
seed_max=1
USER="malhaar-karandikar-na"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python3 ../train/train_smac.py --env_name ${env} --user_name $USER --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 10000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --max_grad_norm 0.5 \
    --entropy_coef 0.001
    
done
