import numpy as np
import re
import os
import pprint

equality = {}
productivity = {}

VIRT_HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/virtue_ethics/layout"
UTIL_HOME_DIR = "/home/mhelabd/ai-ethicist/ai_economist/training/rllib/envs/utilitarian/layout"

for HOME_DIR in [VIRT_HOME_DIR, UTIL_HOME_DIR]:
  equality = {}
  productivity = {}
  for d in os.listdir(HOME_DIR):
    if os.path.isfile(os.path.join(HOME_DIR, d)):
      continue
    results_file = os.path.join(HOME_DIR, d, 'predefined_skill/dense_logs/logs_0000000000048000/logfile.txt')
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
    equality[d] = np.mean(curr_eq)
    productivity[d] = np.mean(curr_prod)
  print(HOME_DIR)
  pprint.pprint(equality)
  pprint.pprint(productivity)

    
