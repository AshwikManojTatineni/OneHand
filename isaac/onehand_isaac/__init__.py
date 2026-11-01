"""OneHand Isaac Lab tasks. Importing this package registers them with gymnasium."""
import gymnasium as gym

gym.register(
    id="OneHand-Franka-PickPlacePost-v0",
    entry_point=f"{__name__}.pick_place_post_env:PickPlacePostEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_post_env:PickPlacePostEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
