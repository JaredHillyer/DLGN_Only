python runfiles/make_runfiles.py
sh runfiles/run.sh

srun -p gpu_p --gres=gpu:1 --qos=gpu_priority  --pty bash