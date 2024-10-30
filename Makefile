get_data:
	python3 data_train/gen_common.py
	python3 data_train/gen_onestop.py
	python3 data_train/gen_cefr.py
	python3 merge_train.py