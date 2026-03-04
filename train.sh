GPU=8
OMP_NUM_THREADS=${GPU} torchrun --master_port 25358 --nproc_per_node=${GPU} matanyone/train.py