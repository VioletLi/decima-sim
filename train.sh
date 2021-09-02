python3 -u train.py \
--exec_cap 4332 \
--model_folder ./baseline_models/ \
--result_folder ./baseline_results/ \
--model_save_interval 10 \
--reset_prob 0.05 \
--reset_prob_decay 0.002 \
--reset_prob_min 0.0001 \
--job_num 478 \
--num_ep 50 \
| tee baseline_train.log