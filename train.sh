python3 -u train.py \
--model_save_interval 10 \
--max_depth 2  \
--hid_dims 4 4 \
--output_dim 4 \
--reset_prob 0.005 \
--reset_prob_decay 0.0002 \
--reset_prob_min 0.00001 \
--num_ep 50 \
--query_type tianchi \
--job_num 478 \
| tee train.log