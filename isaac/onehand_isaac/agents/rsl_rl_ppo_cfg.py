"""PPO settings (rsl_rl, Isaac Lab's default trainer). Sized to match the MJX run's budget: ~150M steps."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500              # 4096 envs x 24 steps x 1500 = 147M steps
    save_interval = 100
    experiment_name = "onehand_franka_pick_place_post"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[256, 256, 256],
                                    critic_hidden_dims=[256, 256, 256], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                     entropy_coef=0.006, num_learning_epochs=5, num_mini_batches=4,
                                     learning_rate=1e-3, schedule="adaptive", gamma=0.98, lam=0.95,
                                     desired_kl=0.01, max_grad_norm=1.0)
