set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

GPU_NUM=8
train_files="['YOUR TRAINING DATASET PATH']"
test_files="['YOUR TEST DATASET PATH']"
project_name='YOUR PROJECT NAME'
experiment_name='YOUR EXPERIMENT NAME'

mkdir -p data-log/$project_name

model_path="YOUR MODEL PATH"


python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.val_batch_size=8 \
    data.max_prompt_length=1536 \
    data.max_response_length=2560 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.use_lookahead=False \
    +actor_rollout_ref.actor.lookahead_inner_steps=3 \
    +actor_rollout_ref.actor.lookahead_step_size=0.8 \
    +actor_rollout_ref.actor.zscore_w=30 \
    +actor_rollout_ref.actor.zscore_threshold=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    +trainer.val_before_train=True \
    +trainer.max_steps=400 \
    trainer.total_epochs=10 \
    | tee data-log/$project_name/$experiment_name.log