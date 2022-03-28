import numpy as np
import re
import os
import pprint

equality = {}
productivity = {}

# HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/virtue_ethics/layout"
HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/utilitarian/layout"

for d in os.listdir(HOME_DIR):
  if os.path.isfile(os.path.join(HOME_DIR, d)):
    continue
  results_file = os.path.join(HOME_DIR, d, 'dense_logs/logs_0000000002406000/logfile.txt')
  if not os.path.isfile(results_file):
    continue
  curr_eq = []
  curr_prod = []
  with open(results_file) as f:
    results = f.read()
    for m in re.finditer('equality:', results):
      curr_eq.append(float(results[m.end(): results.find('\n', m.end())]))
    for m in re.finditer('Productivity:', results):
      curr_prod.append(float(results[m.end(): results.find('\n', m.end())]))
  equality[d] = np.std(curr_eq)
  productivity[d] = np.std(curr_prod)
pprint.pprint(equality)
pprint.pprint(productivity)



    
