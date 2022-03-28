# make envs with layout files for bash script to run
import argparse
import os
import yaml
import numpy as np

def get_agent_morality(num_moral_agents, agent_morality, num_agents=4.):
	moral_agents = agent_morality * np.ones(num_moral_agents)
	immoral_agents = np.zeros(int(num_agents) - len(moral_agents))
	return list([float(i) for i in np.concatenate((immoral_agents, moral_agents), axis=0)])

def make_env(
	morality, 
	num_moral_agents, 
	agent_morality, 
	run_dir,
	random_layout=False, 
	num_agents=4,
	predefined_skill=False, 
):
	morality = 'utilitarian' if morality[0].lower() == 'u' else 'virtue_ethics' if morality[0].lower() == 'v' else 'amoral'
	random_layout = 'random' if random_layout == 1 else 'layout'
	config_path = os.path.join(run_dir, morality, random_layout, 'config.yaml')
	
	with open(config_path, 'r') as f:
		run_configuration = yaml.safe_load(f)

	agent_morality = get_agent_morality(num_moral_agents, agent_morality)
	print('agent_morality: ', agent_morality)
	run_configuration['env']['agent_morality'] = agent_morality
	run_configuration['env']['n_agents'] = num_agents
	agent_morality = [str(i) for i in agent_morality]
	new_path = os.path.join(run_dir, morality, random_layout, 'agent_morality=[' + ','.join(agent_morality) + ']')

	if predefined_skill: # Loads model with defined skill level
		run_configuration['env']['components'][0]['Build']['skill_dist'] = 'predefined'
		run_configuration['general']['episodes'] = 1 # No need for training
		run_configuration['general']['restore_tf_weights_agents'] = os.path.join(new_path, 'ckpts/agent.tf.weights.global-step-2508000')
		new_path = os.path.join(new_path, 'predefined_skill')

	os.makedirs(new_path, exist_ok=True)
	new_config_path = os.path.join(new_path, 'config.yaml')

	with open(new_config_path, 'w+') as f:
		yaml.dump(run_configuration, f,  default_flow_style=False)





## SETUP PARSER
parser = argparse.ArgumentParser(description='Make running environments.')
parser.add_argument('-m', '--morality', type=str, help='type of morality: {[u]tilitarian, [a]moral, [v]irtue_ethics}', default='utilitarian')
parser.add_argument('-n', '--num_agents', type=int, help='Number of agents', default=4)
parser.add_argument('-am', '--agents_morality', type=float, nargs='+', help='list of morality coef: [0, \inf]', default=[1])
parser.add_argument('-r', '--random_layout', type=int, help='whether random or layout', default=0)
parser.add_argument('--run-dir', type=str, help='Path to the directory for this run.', default='envs/')
parser.add_argument('--predefined_skill', action='store_true', help='Make env with predefined building skill.')

args = parser.parse_args()
print(args)
for num_moral_agents in range(args.num_agents + 1):
	for agent_morality in args.agents_morality:
		make_env(
			args.morality, 
			num_moral_agents, 
			agent_morality, 
			args.run_dir,
			random_layout=args.random_layout, 
			num_agents=args.num_agents,
			predefined_skill=args.predefined_skill,
		)

