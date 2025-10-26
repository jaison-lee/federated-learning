echo "=> Generate data.."

cd data/ || exit

rm -r mnist/all_data

python generate_data.py \
  --dataset mnist \
  --n_clients 10 \
  --iid \
  --frac 0.2 \
  --save_dir mnist \
  --seed 1234 \
  # --n_classes_per_client 2 \


cd ../

echo "=> Train.."

python train.py \
  --experiment "mnist" \
  --n_rounds 100 \
  --local_steps 1 \
  --local_optimizer sgd \
  --local_lr 0.001 \
  --server_optimizer sgd \
  --server_lr 0.1 \
  --bz 128 \
  --device "mps" \
  --log_freq 1 \
  --verbose 1 \
  --logs_dir "logs/mnist/test/le1_non_iid_cs2" \
  --seed 12 \
  # --mu 0
  # --sampling_rate 0.2 \
  # --sample_with_replacement
