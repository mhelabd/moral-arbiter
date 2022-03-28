#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2021, salesforce.com, inc.  
# All rights reserved.  
# SPDX-License-Identifier: BSD-3-Clause  
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# ### Colab
# 
# Try this notebook on [Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/multi_agent_training_with_rllib.ipynb).

# ### Prerequisites
# It is helpful to be familiar with **Foundation**, a multi-agent economic simulator built for the AI Economist ([paper here](https://arxiv.org/abs/2004.13332)). If you haven't worked with Foundation before, we highly recommend taking a look at our other tutorials:
# 
# - [Foundation: the Basics](https://github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb)
# - [Extending Foundation](https://github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_advanced.ipynb)
# - [Optimal Taxation Theory and Simulation](https://github.com/salesforce/ai-economist/blob/master/tutorials/optimal_taxation_theory_and_simulation.ipynb)

# ## Introduction

# Welcome! This tutorial is the first of a series on doing distributed multi-agent reinforcement learning (MARL). Here, we specifically demonstrate how to integrate our multi-agent economic simulation, [Foundation](https://github.com/salesforce/ai-economist/tree/master/ai_economist/foundation), with [RLlib](https://github.com/ray-project/ray/tree/master/rllib), an open-source library for reinforcement learning. We chose to use RLlib, as it provides an easy-to-use and flexible library for MARL. A detailed documentation on RLlib is available [here](https://docs.ray.io/en/master/rllib.html).
# 
# We put together these tutorial notebook with the following key goals in mind:
# - Provide an exposition to MARL. While there are many libraries and references out there for single-agent RL training, MARL training is not discussed as much, and there aren't many multi-agent rl libraries.
# - Provide reference starting code to perform MARL training so the AI Economist community can focus more on building meaningful extensions to Foundation and better-performant algorithms.
# 
# We will cover the following concepts in this tutorial:
# 1. Adding an *environment wrapper* to make the economic simulation compatible with RLlib.
# 2. Creating a *trainer* object that holds the (multi-agent) policies for environment interaction.
# 3. Training all the agents in the economic simulation.
# 4. Generate a rollout using the trainer object and visualize it.

# ### Dependencies:
# You can install the ai-economist package using 
# - the pip package manager OR
# - by cloning the ai-economist package and installing the requirements (we shall use this when running on Colab):

# Install OpenAI Gym to help define the environment's observation and action spaces for use with RLlib.

# Install the `RLlib` reinforcement learning library:
# - First, install TensorFlow
# - Then, install ray[rllib]
# 
# Note: RLlib natively supports TensorFlow (including TensorFlow Eager) as well as PyTorch, but most of its internals are framework agnostic. Here's a relevant [blogpost](https://medium.com/distributed-computing-with-ray/lessons-from-implementing-12-deep-rl-algorithms-in-tf-and-pytorch-1b412009297d) that compares running RLlib algorithms with TF and PyTorch. Overall, TF seems to run a bit faster than PyTorch, in our experience, and we will use that in this notebook.

# In[1]:


# Change directory to the tutorials folder
import os, sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    os.chdir("/content/ai-economist/tutorials")
else:
    os.chdir(os.path.dirname(os.path.abspath("__file__"))
)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# ## 1. Adding an Environment Wrapper 

# We first define a configuration (introduced in [the basics tutorial](https://github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basics.ipynb)) for the "gather-trade-build" environment with multiple mobile agents (that move, gather resources, build or trade) and a social planner that sets taxes according to (a scaled variant of) the 2018 US tax schedule.

# In[2]:


lambd = 1
ethics = 'amoral'


# In[3]:


# Define a configuration (dictionary) for the "gather-trade-build" environment.

env_config_dict = {
    # ===== SCENARIO CLASS =====
    # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
    # The environment object will be an instance of the Scenario class.
    'scenario_name': 'moral_uniform/simple_wood_and_stone',
#         ===== MORALITY =====
    'moral_theory': ethics, 
    # 'agent_morality': lambd, 
#     'scenario_name': 'uniform/simple_wood_and_stone',
    
    
    
    # ===== COMPONENTS =====
    # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
    #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
    #   {component_kwargs} is a dictionary of kwargs passed to the Component class
    # The order in which components reset, step, and generate obs follows their listed order below.
    'components': [
        # (1) Building houses
        ('Build', {
            'skill_dist':                   'pareto', 
            'payment_max_skill_multiplier': 3,
            'build_labor':                  10,
            'payment':                      10
        }),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {
            'max_bid_ask':    10,
            'order_labor':    0.25,
            'max_num_orders': 5,
            'order_duration': 50
        }),
        # (3) Movement and resource collection
        ('Gather', {
            'move_labor':    1,
            'collect_labor': 1,
            'skill_dist':    'pareto'
        }),
#         (4) Stealing,
        ('Steal', {
            'steal_labor':    1,
            'skill_dist':    'pareto'  
        }),
    ],
    
    
    # ===== SCENARIO CLASS ARGUMENTS =====
    # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
    'starting_agent_coin': 10,
    
    # ===== STANDARD ARGUMENTS ======
    # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
    'n_agents': 4,          # Number of non-planner agents (must be > 1)
    'world_size': [25, 25], # [Height, Width] of the env world
    'episode_length': 1000, # Number of timesteps per episode
    
    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    'multi_action_mode_agents': True,
    'multi_action_mode_planner': False,
    
    # When flattening observations, concatenate scalar & vector observations before output.
    # Otherwise, return observations with minimal processing.
    'flatten_observations': True,
    # When Flattening masks, concatenate each action subspace mask into a single array.
    # Note: flatten_masks = True is required for masking action logits in the code below.
    'flatten_masks': True,
    
    # How often to save the dense logs
    'dense_log_frequency': 1
}


# Like we have seen in earlier [tutorials](https://github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb), using `env = foundation.make_env_instance(**env_config)` creates an environment instance `env` with the specified configuration.
# 
# In order to use this environment with RLlib, we will also need to add the environment's `observation_space` and `action_space` attributes. Additionally, the environment itself must subclass the [`MultiAgentEnv`](https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py) interface, which can return observations and rewards from multiple ready agents per step. To this end, we use an environment [wrapper](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/env_wrapper.py).

# In[4]:


from rllib.env_wrapper import RLlibEnvWrapper
env_obj = RLlibEnvWrapper({"env_config_dict": env_config_dict}, verbose=True)


# Upon applying the wrapper to our environment, we have now defined observation and action spaces for the agents and the planner, indicated with `(a)` and `(p)` respectively. Also, (a useful tip) you can still access the environment instance and its attributes simply by using `env_obj.env`
# 
# In summary, the observation spaces are represented as `Box` objects and the action spaces as `Discrete` objects (for more details on these types, see the OpenAI documentation [page](https://gym.openai.com/docs/#spaces)).
# 
# Briefly looking at the shapes of the observation features (the numbers in parentheses), you will see that we have some one-dimensional features (e.g. `action-mask`, `flat`, `time`) as well as spatial features (e.g., `world-idx-map`, `world-map`)
# 
# A couple of quick notes:
# - An `action_mask` is used to mask out the actions that are not allowed by the environment. For instance, a mobile agent cannot move beyond the boundary of the world. Hence, in position (0, 0), a mobile cannot move "Left" or "Down", and the corresponding actions in the mask would be nulled out. Now, the RL agent can still recommend to move "Left" or "Down", but the action isn't really taken.
# - The key `flat` arises since we set `flatten_observations': True`. Accordingly, the scalar and vector raw observations are all concatenated into this single key. If you're curious to see the entire set of raw observations, do set `flatten_observations': False` in the env_config, and re-run the above cell.
# 
# Looking at the action spaces, the mobile agents can take 50 possible actions (including 1 NO-OP action or do nothing (always indexed 0), 44 trading-related actions, 4 move actions along the four directions and 1 build action)
# 
# The planner sets the tax rates for 7 brackets, each from 0-100% in steps of 5%, so that's 21 values. Adding the NO-OP action brings the planner action space to `MultiDiscrete([22 22 22 22 22 22 22])`.

# ## 2. Creating a *Trainer* Object

# In order to train our economic simulation environment with RLlib, you will need familiarity with one of the key classes: the [`Trainer`](https://docs.ray.io/en/master/rllib-training.html). The trainer object maintains the relationships that connect each agent in the environment to its corresponding trainable policy, and essentially helps in training, checkpointing policies and inferring actions. It helps to co-ordinate the workflow of collecting rollouts and optimizing the various policies via a reinforcement learning algorithm. Inherently, RLlib maintains a wide suite of [algorithms](https://docs.ray.io/en/master/rllib-algorithms.html) for multi-agent learning (which was another strong reason for us to consider using RLlib) - available options include SAC, PPO, PG, A2C, A3C, IMPALA, ES, DDPG, DQN, MARWIL, APEX, and APEX_DDPG. For the remainder of this tutorial, we will stick to using [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO), an algorithm known to perform well generally.
# 
# Every algorithm has a corresponding trainer object; in the context of PPO, we invoke the `PPOTrainer` object.

# In[5]:


import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer


# PPOTrainer can be instantiated with 
# - `env`: an environment creator (i.e, RLlibEnvWrapper() in our case)
# - `config`: algorithm-specific configuration data for setting the various components of the RL training loop including the environment, rollout worker processes, training resources, degree of parallelism, framework used, and the policy exploration strategies.
# 
# Note: There are several configuration settings related to policy architectures, rollout collection, minibatching, and other important hyperparameters, that need to be set carefully in order to train effectively. For the sake of the high-level exposition, we allow RLlib to use most of the the default settings. Check out the list of default [common configuration parameters](https://docs.ray.io/en/releases-0.8.4/rllib-training.html#common-parameters) and default [PPO-specific configuration parameters](https://docs.ray.io/en/releases-0.8.4/rllib-algorithms.html?highlight=PPO#proximal-policy-optimization-ppo). Custom environment configurations may be passed to environment creator via `config["env_config"]`.
# 
# RLlib also chooses default built-in [models](https://docs.ray.io/en/releases-0.8.4/rllib-models.html#built-in-models-and-preprocessors) for processing the observations. The models are picked based on a simple heuristic: a [vision](https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet.py) network for observations that have shape of length larger than 2 (for example, (84 x 84 x 3)), and a [fully connected](https://github.com/ray-project/ray/blob/master/rllib/models/tf/fcnet.py) network for everything else. Custom models can be configured via the `config["policy"]["model"]` key.
# 
# In the context of multi-agent training, we will also need to set the multi-agent configuration:
# ```python
# "multiagent": {
#         # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
#         # of (policy_cls, obs_space, act_space, config). This defines the
#         # observation and action spaces of the policies and any extra config.
#         "policies": {},
#         # Function mapping agent ids to policy ids.
#         "policy_mapping_fn": None,
#         # Optional list of policies to train, or None for all policies.
#         "policies_to_train": None,
#     },
# ```
# 
# To this end, let's notate the agent policy id by `"a"` and the planner policy id by `"p"`. We can set `policies`, `policy_mapping_fun` and `policies_to_train` as follows.

# In[6]:


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

# In foundation, all the agents have integer ids and the social planner has an id of "p"
policy_mapping_fun = lambda i: "a" if str(i).isdigit() else "p"

policies_to_train = ["a", "p"]


# Create a multiagent trainer config holding the trainable policies and their mappings.

# In[7]:


trainer_config = {
    "multiagent": {
        "policies": policies,
        "policies_to_train": policies_to_train,
        "policy_mapping_fn": policy_mapping_fun,
    }
}


# With distributed RL, architectures typically comprise several **roll-out** and **trainer** workers operating in tandem
# ![](assets/distributed_rl_architecture.png)
# 
# The roll-out workers repeatedly step through the environment to generate and collect roll-outs in parallel, using the actions sampled from the policy models on the roll-out workers or provided by the trainer worker.
# Roll-out workers typically use CPU machines, and sometimes, GPU machines for richer environments.
# Trainer workers gather the roll-out data (asynchronously) from the roll-out workers and optimize policies on CPU or GPU machines.
# 
# In this context, we can also add a `num_workers` configuration parameter to specify the number of rollout workers, i.e, those responsible for gathering rollouts. Note: setting `num_workers=0` will mean the rollouts will be collected by the trainer worker itself. Also, each worker can collect rollouts from multiple environments in parallel, which is specified in `num_envs_per_worker`; there will be a total of `num_workers` $\times$ `num_envs_per_worker` environment replicas used to gather rollouts.
# Note: below, we also update some of the default trainer settings to keep the iteration time small.

# In[8]:


trainer_config.update(
    {
        "num_workers": 2,
        "num_envs_per_worker": 2,
        # Other training parameters
        "train_batch_size":  4000,
        "sgd_minibatch_size": 4000,
        "num_sgd_iter": 1
    }
)


# Finally, we need to add the environment configuration to the trainer configuration.

# In[9]:


# We also add the "num_envs_per_worker" parameter for the env. wrapper to index the environments.
env_config = {
    "env_config_dict": env_config_dict,
    "num_envs_per_worker": trainer_config.get('num_envs_per_worker'),   
}

trainer_config.update(
    {
        "env_config": env_config, 
    }
)


# One the training configuration is set, we will need to initialize ray and create the PPOTrainer object.

# In[10]:


# Initialize Ray
ray.init()


# In[11]:


# Create the PPO trainer.
# TODO: MAKE Q-LEARNING 
trainer = PPOTrainer(
    env=RLlibEnvWrapper,
    config=trainer_config,
    )


# ## 3. Perform Training

# And that's it! We are now ready to perform training by invoking `trainer.train()`; we call it for just a few number of iterations.

# In[ ]:

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())

NUM_ITERS = 2000
for iteration in range(NUM_ITERS):
    print(f'********** Iter : {iteration} **********')
    result = trainer.train()
    print(f'''episode_reward_mean: {result.get('episode_reward_mean')}''')


# By default, the results will be logged to a subdirectory of `~/ray_results`. This subdirectory will contain a file `params.json` which contains the hyperparameters, a file `result.json` which contains a training summary for each episode and a TensorBoard file that can be used to visualize training process with TensorBoard by running|
# ```shell
# tensorboard --logdir ~/ray_results
# ```

# ## 4. Generate and Visualize the Environment's Dense Logs

# At any point during training, we would also want to inspect the environment's dense logs in order to deep-dive into the training results. Introduced in our [basic tutorial](https://github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb#Visualize-using-dense-logging), dense logs are basically logs of each agent's states, actions and rewards at every point in time, along with a snapshot of the world state.
# 
# There are two equivalent ways to fetch the environment's dense logs using the trainer object.
# 
# a. Simply retrieve the dense log from the workers' environment objects
# 
# b. Generate dense log(s) from the most recent trainer policy model weights

# ### 4a. Simply retrieve the dense log from the workers' environment objects

# From each rollout worker, it's straightforward to retrieve the dense logs using some of the function attributes.

# In[ ]:


# Below, we fetch the dense logs for each rollout worker and environment within

dense_logs = {}
# Note: worker 0 is reserved for the trainer actor
for worker in range((trainer_config["num_workers"] > 0), trainer_config["num_workers"] + 1):
    for env_id in range(trainer_config["num_envs_per_worker"]):
        dense_logs["worker={};env_id={}".format(worker, env_id)] =         trainer.workers.foreach_worker(lambda w: w.async_env)[worker].envs[env_id].env.previous_episode_dense_log


# In[ ]:


# We should have num_workers x num_envs_per_worker number of dense logs
print(dense_logs.keys())


# ### 4b. Generate a dense log from the most recent trainer policy model weights

# We may also use the trainer object directly to play out an episode. The advantage of this approach is that we can re-sample the policy model any number of times and generate several rollouts.

# In[ ]:


def generate_rollout_from_current_trainer_policy(
    trainer, 
    env_obj,
    num_dense_logs=1,
    plot_every=10
):
    dense_logs = {}
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for idx in range(num_dense_logs):
        # Set initial states
        agent_states = {}
        for agent_idx in range(env_obj.env.n_agents):
            agent_states[str(agent_idx)] = trainer.get_policy("a").get_initial_state()
        planner_states = trainer.get_policy("p").get_initial_state()   

        # Play out the episode
        obs = env_obj.reset(force_dense_logging=True)
        for t in range(env_obj.env.episode_length):
            actions = {}
            for agent_idx in range(env_obj.env.n_agents):
                # Use the trainer object directly to sample actions for each agent
                actions[str(agent_idx)] = trainer.compute_action(
                    obs[str(agent_idx)], 
                    agent_states[str(agent_idx)], 
                    policy_id="a",
                    full_fetch=False
                )

            # Action sampling for the planner
            actions["p"] = trainer.compute_action(
                obs['p'], 
                planner_states, 
                policy_id='p',
                full_fetch=False
            )

            obs, rew, done, info = env_obj.step(actions)        
            if done['__all__']:
                break
        dense_logs[idx] = env_obj.env.dense_log   
    return dense_logs


# In[ ]:


dense_logs = generate_rollout_from_current_trainer_policy(
    trainer, 
    env_obj,
    num_dense_logs=4
)


# ### Visualizing the episode dense logs

# Once we obtain the dense logs, we can use the plotting utilities we have created to examine the episode dense logs and visualize the the world state, agent-wise quantities, movement, and trading events.

# In[ ]:


from utils import plotting  # plotting utilities for visualizing env. state

dense_log_idx = len(dense_logs)-1
plotting.breakdown(dense_logs[dense_log_idx]);


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from ai_economist.foundation.scenarios.utils import social_metrics


def do_plot(env, ax, fig):
    """Plots world state during episode sampling."""
    plotting.plot_env_state(env, ax)
    ax.set_aspect('equal')
    display.display(fig)
    display.clear_output(wait=True)

def play_random_episode(env, plot_every=100, do_dense_logging=False):
    """Plays an episode with randomly sampled actions.
    
    Demonstrates gym-style API:
        obs                  <-- env.reset(...)         # Reset
        obs, rew, done, info <-- env.step(actions, ...) # Interaction loop
    
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Reset
    obs = env.reset(force_dense_logging=do_dense_logging)
    agent_states = {}
    for agent_idx in range(env_obj.env.n_agents):
        agent_states[str(agent_idx)] = trainer.get_policy("a").get_initial_state()
    planner_states = trainer.get_policy("p").get_initial_state()   


    for t in range(env_obj.env.episode_length):
        actions = {}
        for agent_idx in range(env_obj.env.n_agents):
            # Use the trainer object directly to sample actions for each agent
            actions[str(agent_idx)] = trainer.compute_action(
                obs[str(agent_idx)], 
                agent_states[str(agent_idx)], 
                policy_id="a",
                full_fetch=False
            )

        # Action sampling for the planner
        actions["p"] = trainer.compute_action(
            obs['p'], 
            planner_states, 
            policy_id='p',
            full_fetch=False
        )

        obs, rew, done, info = env_obj.step(actions)  

        if ((t+1) % plot_every) == 0:
            do_plot(env.env, ax, fig)

        if ((t+1) % plot_every) != 0:
            do_plot(env.env, ax, fig) 
    if do_dense_logging:
        plotting.breakdown(env_obj.env.dense_log)
    coin_endowments = np.array([agent.total_endowment("Coin") for agent in env_obj.env.world.agents])
    print("equality: {}".format(social_metrics.get_equality(coin_endowments)))
    print("Productivity: {}".format(social_metrics.get_productivity(coin_endowments)))
    print("lambda: {}".format(lambd))
    print("ethic: {}".format(ethics))


# In[ ]:


coin_endowments = np.array([agent.total_endowment("Coin") for agent in env_obj.env.world.agents])
print("equality: {}".format(social_metrics.get_equality(coin_endowments)))
print("Productivity: {}".format(social_metrics.get_productivity(coin_endowments)))
print("lambda: {}".format(lambd))
print("ethic: {}".format(ethics))


# In[ ]:





# In[ ]:


# Shutdown Ray after use
ray.shutdown()


# And that's it for now. See you in the [next](https://github.com/salesforce/ai-economist/blob/master/tutorials/two_level_curriculum_learning_with_rllib.md) tutorial :)
