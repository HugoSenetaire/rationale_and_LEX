python -u scripts/main.py  --batch_size 64--dataset news_group --embedding \
glove --dropout 0.05 --weight_decay 5e-06 --num_layers 1 --model_form cnn --hidden_dim 100 --epochs 50 --init_lr 0.0001 --num_workers \
 0 --objective cross_entropy --patience 5 --save_dir snapshot --train --test --results_path logs/demo_run.results  --gumbel_decay 1e-5 --get_rationales \
 --selection_lambda .001 --continuity_lambda 0

