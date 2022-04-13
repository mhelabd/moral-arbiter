cd ~/ai-ethicist/
python ./ai_economist/training/rllib/make_envs.py -m u --run-dir ./ai_economist/training/rllib/envs/ -am 0.1 1 10   --predefined_skill
python ./ai_economist/training/rllib/make_envs.py -m v --run-dir ./ai_economist/training/rllib/envs/ -am 0.1  10  1000  --predefined_skill