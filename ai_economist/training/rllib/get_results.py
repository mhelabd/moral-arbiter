import numpy as np
import re
import os
import pprint
import numpy as np
import pandas as pd

equality = {}
productivity = {}

VIRT_HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/virtue_ethics/layout/"
UTIL_HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/utilitarian/layout/"
AI_HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/AI/layout/phase2/"

writer = pd.ExcelWriter('results.xlsx')

for ethic, HOME_DIR in {'virtue_ethics': VIRT_HOME_DIR, 'utilitarian': UTIL_HOME_DIR, 'AI': AI_HOME_DIR}.items():
  equality = {}
  productivity = {}
  eq_times_prod = {}
  for d in os.listdir(HOME_DIR):
    if os.path.isfile(os.path.join(HOME_DIR, d)):
      continue
    results_folder = os.path.join(HOME_DIR, d, 'predefined_skill/dense_logs/')
    try:
      results_file = os.path.join(results_folder, os.listdir(results_folder)[0], 'logfile.txt')
    except:
      print('FILE DOES NOT EXIST', results_folder)
      continue
    if not os.path.isfile(results_file):
      continue
    curr_eq = []
    curr_prod = []
    curr_eq_times_prod = []
    with open(results_file) as f:
      results = f.read()
      for m in re.finditer('equality:', results):
        curr_eq.append(float(results[m.end(): results.find('\n', m.end())]))
      for m in re.finditer('Productivity:', results):
        curr_prod.append(float(results[m.end(): results.find('\n', m.end())]))
        curr_eq_times_prod.append(curr_eq[-1] * curr_prod[-1] / 4)
    equality[d] = np.median(curr_eq)
    productivity[d] = np.median(curr_prod)
    eq_times_prod[d] = np.median(curr_eq_times_prod)
  print(HOME_DIR)
  morality_coef_to_idx = {}
  equality_keys = list(equality.keys())
  equality_keys.sort()
  for k in equality_keys:
    og_k = k
    k = k[len('agent_morality=['):-1].split(',')
    k = [float(i) for i in k]
    morality_coef = k[0] # last one has the moral
    morality_coef_to_idx[morality_coef] = morality_coef_to_idx.get(morality_coef, len(morality_coef_to_idx.keys()))
  equality_np = np.zeros((5, len(morality_coef_to_idx.keys())))
  productivity_np = np.zeros((5, len(morality_coef_to_idx.keys())))
  eq_times_prod_np = np.zeros((5, len(morality_coef_to_idx.keys())))

  for k, v in equality.items():
    og_k = k
    k = k[len('agent_morality=['):-1].split(',')
    k = [float(i) for i in k]
    num_moral_agents = (4 - np.sum(np.array(k) == 0.0))
    morality_coef = k[0] # last one has the moral
    
    equality_np[num_moral_agents][morality_coef_to_idx[morality_coef]] = v
    productivity_np[num_moral_agents][morality_coef_to_idx[morality_coef]] = productivity[og_k]
    eq_times_prod_np[num_moral_agents][morality_coef_to_idx[morality_coef]] = eq_times_prod[og_k]
  
  for name, result in {'equality': equality_np.T, 'productivity': productivity_np.T, 'eq_times_prod': eq_times_prod_np.T}.items():
    df = pd.DataFrame(result)
    # Write each dataframe to a different worksheet.
    df.columns.name = '# Moral Agents'
    df.index = list(morality_coef_to_idx.keys())
    df.index.name = 'Moral Coef \ # Moral Agents'
    df.columns = np.arange(equality_np.shape[0])
    df.reset_index(inplace=True)
    print(df)
    df.to_excel(writer, sheet_name=ethic + '_' + name)
writer.save()

  # for d in [equality_dict, productivity_dict, eq_times_prod_dict]:
  #   print(np.array(d.values()))

  # equality_np = np.zeros((len(equality)//4, 4))
  # productivity_np = np.zeros((len(equality)//4, 4))
  # eq_times_prod_np = np.zeros((len(equality)//4, 4))
