import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
from ai_economist.foundation.scenarios.utils import social_metrics
from tutorials.utils import plotting
from os import makedirs
from os import listdir
from os.path import isfile, join
import cv2


def get_logs(trainer, env, basedir=None, moral_theory="None", agent_morality=0):
    dense_logs = {}

    # Note: worker 0 is reserved for the trainer actor
    trainer_config = trainer.get_config()
    for worker in range((trainer_config["num_workers"] > 0), trainer_config["num_workers"] + 1):
        for env_id in range(trainer_config["num_envs_per_worker"]):
            dense_logs["worker={};env_id={}".format(worker, env_id)] = \
                trainer.workers.foreach_worker(lambda w: w.async_env)[
                worker].envs[env_id].env.previous_episode_dense_log
    for k, v in dense_logs.items():
        if v.get('world', []) == []: #TODO: fix this
            continue
        (fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(v)
        if basedir is not None:
            fig0.savefig(f'{basedir}{k}, 0.png')
            fig1.savefig(f'{basedir}{k}, 1.png')
            fig2.savefig(f'{basedir}{k}, 2.png')
        endows = np.array(endows)
        print("coin endowments: {}".format(','.join([str(i) for i in endows])))
        print("equality: {}".format(social_metrics.get_equality(endows)))
        print("Productivity: {}".format(social_metrics.get_productivity(endows)))
        print("Moral Theory: {}".format(moral_theory))
        print("Agent Morality: {}".format(agent_morality))
        print("Learned Morality: {}".format(env.env.world.planner.state.get('curr_moral_values', 'None')))
    return dense_logs


def do_plot(env, ax, fig, i, basedir):
    """Plots world state during episode sampling."""
    plotting.plot_env_state(env, ax)
    ax.set_aspect('equal')
    if basedir:
        fig.savefig(f'{basedir}/{i}.png') # TODO fix plotting
    plt.cla()


def play_random_episode(
    trainer, env_obj, plot_every=20, do_dense_logging=False,
    basedir=None, moral_theory='amoral', agent_morality=0
):
    """Plays an episode with randomly sampled actions.

    Demonstrates gym-style API:
        obs                  <-- env.reset(...)         # Reset
        obs, rew, done, info <-- env.step(actions, ...) # Interaction loop

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Reset
    obs = env_obj.reset(force_dense_logging=do_dense_logging)
    agent_states = {}
    for agent_idx in range(env_obj.env.n_agents):
        agent_states[str(agent_idx)] = trainer.get_policy(
            "a").get_initial_state()
    planner_states = trainer.get_policy("p").get_initial_state()
    if basedir:
        mediadir = f'{basedir}/media'
        makedirs(mediadir, exist_ok=True)
    for t in range(env_obj.env.episode_length):
        actions = {}
        for agent_idx in range(env_obj.env.n_agents):
            # Use the trainer object directly to sample actions for each agent
            actions[str(agent_idx)], _, _ = trainer.compute_action(
                obs[str(agent_idx)], 
                agent_states[str(agent_idx)], 
                policy_id='a',
            )

        # Action sampling for the planner
        actions["p"] = trainer.compute_action(
            obs['p'], 
            planner_states, 
            policy_id='p',
        )

        obs, rew, done, info = env_obj.step(actions)

        if ((t+1) % plot_every) == 0:
            do_plot(env_obj.env, ax, fig, t, mediadir)
    plt.close('all')

    if do_dense_logging:
        logdir = join(basedir, 'denselogs/')
        makedirs(logdir, exist_ok=True)
        get_logs(trainer, env_obj, logdir, getattr(env_obj.env, '_moral_theory', "None"), getattr(env_obj.env, '_agent_morality', 0))
    if basedir:
        filenames = [f for f in listdir(mediadir) if isfile(
            join(mediadir, f)) and f.endswith(".png")]
        frame = cv2.imread(join(mediadir, filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(join(mediadir, 'video.mp4'), cv2.VideoWriter_fourcc(
            *'mp4v'), 20.0, (width, height))
        for image in filenames:
            video.write(cv2.imread(join(mediadir, image)))
