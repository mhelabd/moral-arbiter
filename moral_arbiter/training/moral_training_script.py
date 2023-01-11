# Usage:
# >> python moral_arbiter/training/moral_training_script.py --env moral_economy

import argparse
import os
import GPUtil
import ray
import plots
import sys
original_stdout = sys.stdout 

try:
    num_gpus_available = len(GPUtil.getAvailable())
    assert num_gpus_available > 0, "This training script needs a GPU to run!"
    print(f"Inside moral_training_script.py: {num_gpus_available} GPUs are available.")
    import yaml
    from ray.rllib.agents.ppo import PPOTrainer
    from rllib.env_wrapper import RLlibEnvWrapper
except ValueError:
    raise ValueError("This training script needs a GPU to run!") from None

MORAL_ECONOMY = "moral_economy"
SIMPLE_WOOD_AND_STONE = "simple_wood_and_stone"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str, help="Environment to train.")
    parser.add_argument("--morality", "-m", type=str, help="Moral Framework")

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
        f"{args.morality}.yaml",
    )
    with open(config_path, "r", encoding="utf8") as f:
        run_config = yaml.safe_load(f)

    saving_config = run_config['saving']
    os.makedirs(saving_config['basedir'] + saving_config['name'], exist_ok=True)
    log_file = open(saving_config['basedir'] + saving_config['name'] + "/logfile.txt", "w+")
    sys.stdout = log_file

    ray.init()

    env_obj = RLlibEnvWrapper({"env_config_dict": run_config['env']}, verbose=True)

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
    policies_to_train = ["a"]

    trainer_config = {
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fun,
        }, 
        "env_config": { 
            "env_config_dict": run_config['env'],
            "num_envs_per_worker": run_config['trainer']['num_envs_per_worker'],   
        }
    }
    trainer_config.update(run_config['trainer'])

    trainer = PPOTrainer(
        env=RLlibEnvWrapper,
        config=trainer_config,
    )

    NUM_ITERS = 5000
    for i in range(NUM_ITERS):
        print(f'********** Iter : {i} **********')
        result = trainer.train()
        print(f'''episode_reward_mean: {result.get('episode_reward_mean')}''')
        if i % saving_config['model_params_save_freq'] == 0 or i==NUM_ITERS-1:
            checkpoint = trainer.save(saving_config['basedir'] + saving_config['name'] + '/weights/')
            checkpoint = checkpoint[:checkpoint.rfind('/')+1]
            statsdir = checkpoint.replace('weights', 'stats')
            os.makedirs(statsdir, exist_ok=True)
            with open(statsdir + 'stats.txt', 'w+') as f:
                sys.stdout = f
                plots.play_random_episode(trainer, env_obj, do_dense_logging=True, basedir=statsdir)
            sys.stdout = log_file
            print("checkpoint saved at", checkpoint)
    ray.shutdown()
    log_file.close()
