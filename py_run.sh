CUDA_VISIBLE_DEVICES=6 \
nohup python3.8 ./run.py \
--do_train \
--do_eval \
--do_test \
--model_name_or_path microsoft/unixcoder-base-nine \
--train_filename /data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/python/train \
--dev_filename /data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/python/test \
--test_filename /data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/python/test \
--output_dir ./saved_models/final_woYx_4_Bina_contrast_py_avgEmb_trans_copy \
--max_source_length 256 \
--max_path_length 40 \
--max_target_length 128 \
--num_layers 4 \
--beam_size 4 \
--train_batch_size 80 \
--eval_batch_size 80 \
--learning_rate 5e-5 \
--gradient_accumulation_steps 1 \
--num_train_epochs 200 > final_woYx_4_Bina_contrast_py_avgEmb_trans_copy.out
