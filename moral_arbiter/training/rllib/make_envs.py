# make envs with layout files for bash script to run
import argparse
import os
import yaml
import numpy as np
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_agent_morality(num_moral_agents, agent_morality, num_agents=4.):
	moral_agents = agent_morality * np.ones(num_moral_agents)
	immoral_agents = np.zeros(int(num_agents) - len(moral_agents))
	return list([float(i) for i in np.concatenate((moral_agents, immoral_agents), axis=0)])

def make_env(
	morality, 
	num_moral_agents, 
	agent_morality, 
	run_dir,
	random_layout=False, 
	num_agents=4,
	predefined_skill=False, 
	use_gpu=False, 
	visible_device_list=-1, 
):
	m = morality[0].lower()
	if m == 'u':
		morality = 'utilitarian'
	elif m == 'v':
		morality = 'virtue_ethics'
	elif m == 'a':
		morality = 'AI'
	else:
		morality = 'amoral'

	random_layout = 'random' if random_layout == 1 else 'layout'
	random_layout = random_layout if morality != 'AI' else os.path.join(random_layout, 'phase2')
	config_path = os.path.join(run_dir, morality, random_layout, 'config.yaml')
	
	with open(config_path, 'r') as f:
		run_configuration = yaml.safe_load(f)

	agent_morality = get_agent_morality(num_moral_agents, agent_morality)
	print('agent_morality: ', agent_morality)
	if use_gpu:
		run_configuration['trainer']['num_gpus'] = 1
		run_configuration['trainer']['num_workers'] = 12
		run_configuration['general']['cpus'] = 12
		run_configuration['general']['gpus'] = 1
		run_configuration['trainer']['tf_session_args']['device_count']['CPU'] = 12
		run_configuration['trainer']['tf_session_args']['device_count']['GPU'] = 1
		run_configuration['trainer']['tf_session_args']['gpu_options']['visible_device_list'] = visible_device_list
	else:
		run_configuration['general']['cpus'] = 28
		run_configuration['general']['gpus'] = 0
		run_configuration['trainer']['num_workers'] = 28
		run_configuration['trainer']['tf_session_args']['device_count']['CPU'] = 28
		run_configuration['trainer']['tf_session_args']['device_count']['GPU'] = 0

	run_configuration['trainer']['seed'] = 1024
	run_configuration['env']['agent_morality'] = agent_morality
	run_configuration['env']['n_agents'] = num_agents
	agent_morality = [str(i) for i in agent_morality]
	new_path = os.path.join(run_dir, morality, random_layout, 'agent_morality=[' + ','.join(agent_morality) + ']')

	if predefined_skill: # Loads model with defined skill level
		# run_configuration['env']['components'][0]['Build']['skill_dist'] = 'predefined'
		run_configuration['general']['episodes'] = 100 # No need for training
		run_configuration['env']['dense_log_frequency'] = 1000 # Will log at the end
		run_configuration['general']['train_planner'] = morality == 'AI'
		run_configuration['general']['train_agent'] = True
		try:
			results_folder =os.path.join('/home/mhelabd/moral-arbiter', new_path, 'ckpts')
			results_files = [f for f in os.listdir(results_folder) if re.match(r'agent.tf.weights.global-step.*', f)]
			results_files.sort(reverse=True, key=natural_keys)
			run_configuration['general']['restore_tf_weights_agents'] = os.path.join(results_folder, results_files[0])
		except:
			print('NOT TRAINED: ', ('[' + ','.join(agent_morality) + ']'))
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
parser.add_argument('--one_env', action='store_true', help='Make one env with 0.5 morality.')
parser.add_argument('--gpu', action='store_true', help='CPU VS GPU')
parser.add_argument('--visible_device_list', type=str,  help='Which GPU', default='0')


args = parser.parse_args()
print(args)
if args.one_env:
	make_env(
		args.morality, 
		4, 
		[0.5, 0.5, 0.5, 0.5], 
		args.run_dir,
		random_layout=args.random_layout, 
		num_agents=args.num_agents,
		predefined_skill=args.predefined_skill,
		use_gpu=args.gpu,
		visible_device_list=args.visible_device_list, 
	)
	exit(0)
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
			use_gpu=args.gpu,
			visible_device_list=args.visible_device_list, 
		)

