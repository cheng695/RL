from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MyRobotRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the UZ-05 rough velocity task."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "uz05_rough"
    run_name = ""
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MyRobotFlatPPORunnerCfg(MyRobotRoughPPORunnerCfg):
    """RSL-RL PPO configuration for the UZ-05 flat velocity task."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "uz05_flat"
