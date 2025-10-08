echo "=> Generate data.."

cd data/ || exit

rm -r mnist/all_data
rm -r cifar10/all_data

python generate_data.py \
  --dataset cifar10 \
  --n_clients 10 \
  --iid \
  --frac 0.2 \
  --save_dir cifar10 \
  --seed 1234

cd ../

echo "=> Train.."

python train.py \
  --experiment "cifar10" \
  --aggregator_type "centralized" \
  --n_rounds 100 \
  --local_steps 50 \
  --local_optimizer sgd \
  --local_lr 0.001 \
  --server_optimizer sgd \
  --server_lr 0.1 \
  --bz 64 \
  --device "mps" \
  --log_freq 1 \
  --verbose 1 \
  --logs_dir "logs/cifar10/10_bz64" \
  --seed 12
