cd ~/ai-ethicist/
python ./ai_economist/training/rllib/make_envs.py -m u --run-dir ./ai_economist/training/rllib/envs/ -am 0.1 1 10 --gpu --visible_device_list 1
python ./ai_economist/training/rllib/make_envs.py -m v --run-dir ./ai_economist/training/rllib/envs/ -am 0.1  10  1000 --gpu --visible_device_list 2