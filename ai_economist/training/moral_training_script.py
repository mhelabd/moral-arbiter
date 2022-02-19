# Usage:
# >> python ai_economist/training/moral_training_script.py --env moral_economy

import argparse
import os
import GPUtil

from ai_economist.foundation.scenarios import simple_wood_and_stone

try:
    num_gpus_available = len(GPUtil.getAvailable())
    assert num_gpus_available > 0, "This training script needs a GPU to run!"
    print(f"Inside moral_training_script.py: {num_gpus_available} GPUs are available.")
    import yaml
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.env_wrapper import RLlibEnvWrapper
except ValueError:
    raise ValueError("This training script needs a GPU to run!") from None

MORAL_ECONOMY = "moral_economy"
SIMPLE_WOOD_AND_STONE = "simple_wood_and_stone"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str, help="Environment to train.")

    args = parser.parse_args()

    # Read the run configurations specific to each environment.
    # Note: The run config yamls are located at warp_drive/training/run_configs
    # ---------------------------------------------------------------------------
    assert args.env in [MORAL_ECONOMY, SIMPLE_WOOD_AND_STONE], (
        f"Currently, the only environment supported "
        f"is {MORAL_ECONOMY}, {SIMPLE_WOOD_AND_STONE}"
    )

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_configs",
        f"{args.env}.yaml",
    )
    with open(config_path, "r", encoding="utf8") as f:
        run_config = yaml.safe_load(f)

    env_obj = RLlibEnvWrapper({"env_config_dict": run_config}, verbose=True)

    # POLICIES
    policies = {
        "a": (
            None,  # uses default policy
            env_obj.observation_space,
            env_obj.action_space,
            {}  # define a custom agent policy configuration.
        ),
        "p": (
            None,  # uses default policy
            env_obj.observation_space_pl,
            env_obj.action_space_pl,
            {}  # define a custom planner policy configuration.
        )
    }
    policy_mapping_fun = lambda i: "a" if str(i).isdigit() else "p"
    policies_to_train = ["a", "p"]

    trainer_config = {
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fun,
        }, 
        "num_workers": 16,
        "num_envs_per_worker": 4,
        # Other training parameters
        "train_batch_size":  4000,
        "sgd_minibatch_size": 4000,
        "num_sgd_iter": 1000,
        "env_config": { 
            "env_config_dict": run_config,
            "num_envs_per_worker": 4,   
        }
    }
    trainer = PPOTrainer(
        env=RLlibEnvWrapper,
        config=trainer_config,
    )

    trainer.train()
    trainer.graceful_close()
