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
AI_HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/saved/AI/layout/phase2/"

writer = pd.ExcelWriter('income_costs_results_per_unit.xlsx')
# 'virtue_ethics': VIRT_HOME_DIR, 'utilitarian': UTIL_HOME_DIR,
for ethic, HOME_DIR in { 'AI': AI_HOME_DIR}.items():
  income_cost_dict = {} # {d: {cost wood: [# agent 0 cost wood, agent 1 cost wood]}
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
    curr_income_cost_dict = {}
    with open(results_file) as f:
      results = f.read()
      for item in ["Cost \(Wood\)    :", "Cost \(Stone\)   :", "Income \(Wood\)  :", "Income \(Stone\) :", "Income \(Build\) :", "Cost \(Steals\)  :", "Income \(Steals\):"]:
        curr_income_cost_dict[item] = []
        for m in re.finditer(item, results):
          loc = results[m.end(): results.find('\n', m.end())]
          loc = loc.replace('~~~~~~~~', '0.0 (n= 0)')
          result = re.findall(r"\s*([+-]?([0-9]*[.])?[0-9]+)\s*\(n=\s*(\d*)\)", loc)
          result = [float(x[0]) for x in result]
          curr_income_cost_dict[item].append(result)
          # print(curr_income_cost_dict)
    income_cost_dict[d] = {k: np.mean(np.array(v), axis=0) for k, v in curr_income_cost_dict.items()} 

  for k, v in income_cost_dict.items():
    v_df =pd.DataFrame.from_dict(v,orient='index').transpose()

    v_df.columns.name = 'Agent #'
    k = k.replace('[', '(').replace(']', ')')
    k = k[len('agent_morality='):]
    v_df.to_excel(writer, sheet_name=ethic + '_' + k)
writer.save()

#   print(HOME_DIR)
#   morality_coef_to_idx = {}
#   income_cost_dict_keys = list(income_cost_dict.keys())
#   income_cost_dict_keys.sort()
#   for k in income_cost_dict_keys:
#     og_k = k
#     k = k[len('agent_morality=['):-1].split(',')
#     k = [float(i) for i in k]
#     morality_coef = k[0] # last one has the moral
#     morality_coef_to_idx[morality_coef] = morality_coef_to_idx.get(morality_coef, len(morality_coef_to_idx.keys()))
#   income_cost_np = np.zeros((5, len(morality_coef_to_idx.keys())))

#   for k, v in equality.items():
#     og_k = k
#     k = k[len('agent_morality=['):-1].split(',')
#     k = [float(i) for i in k]
#     num_moral_agents = (4 - np.sum(np.array(k) == 0.0))
#     morality_coef = k[0] # last one has the moral
    
#     income_cost_np[num_moral_agents][morality_coef_to_idx[morality_coef]] = v
  
#   for name, result in {'equality': equality_np.T, 'productivity': productivity_np.T, 'eq_times_prod': eq_times_prod_np.T}.items():
#     df = pd.DataFrame(result)
#     # Write each dataframe to a different worksheet.
#     df.columns.name = '# Moral Agents'
#     df.index = list(morality_coef_to_idx.keys())
#     df.index.name = 'Moral Coef \ # Moral Agents'
#     df.columns = np.arange(equality_np.shape[0])
#     df.reset_index(inplace=True)
#     print(df)
#     df.to_excel(writer, sheet_name=ethic + '_' + name)
# writer.save()
