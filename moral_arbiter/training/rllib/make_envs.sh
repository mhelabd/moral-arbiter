cd ~/moral-arbiter/
python ./moral_arbiter/training/rllib/make_envs.py -m u --run-dir ./moral_arbiter/training/rllib/envs/ -am 0.1 1 10 
python ./moral_arbiter/training/rllib/make_envs.py -m v --run-dir ./moral_arbiter/training/rllib/envs/ -am 0.1  10  1000