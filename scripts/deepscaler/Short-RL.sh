MODEL_PATH=Qwen/Qwen2.5-Math-1.5B-Instruct

export VLLM_ATTENTION_BACKEND=XFORMERS

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

ray stop --force
ray start --head

Name=shortRL_qwen1.5B_0911
SavePath=/home/cliu/deepscaler/checkpoints/$Name
basepath="/home/cliu/deepscaler/deepscaler/data/deepscaler"
length_tolerance=100
acc_tolerance=0.05
reward_type=ShortRL
# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$basepath/train.parquet \
    data.val_files=$basepath/math500.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.validate_roll_out_max_length=9216 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='Length-LLM' \
    trainer.experiment_name=$Name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=20 \
    +trainer.save_limit=4 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=5 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=$SavePath \
    +trainer.reward_type=$reward_type \
    +algorithm.acc_tolerance=$acc_tolerance \
    +algorithm.length_tolerance=$length_tolerance