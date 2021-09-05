python3 test.py \
--test_schemes learn \
--exec_cap 4332 \
--model_folder ./baseline_models/ \
--result_folder ./baseline_results/ \
--model_save_interval 10 \
--reset_prob 0.05 \
--reset_prob_decay 0.002 \
--reset_prob_min 0.0001 \
--job_num 478 \
--saved_model ./baseline_models/model_ep_10 \
--max_depth 2 \
--hid_dims 4 4 \
--output_dim 4 \
--query_type tianchi \
| tee baseline_test.log