## CyberQ - AISecKG-QA-Dataset
## Kg-augmented LLMs Prompting for Cybersecurity Education


![pipeline_1](https://github.com/garima0106/AISECKG-QA-Dataset/assets/54346120/1a54788c-8826-47c8-9298-83ac1757032c)


### SQUAD
python qa.py   --model_name_or_path t5-small   --dataset_name squad_v2   --context_column context   --question_column question   --answer_column answers   --do_train   --do_eval   --per_device_train_batch_size 4   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 128   --doc_stride 128   --output_dir ./outputs   --overwrite_output_dir 

### OURS
python qa.py \
  --model_name_or_path t5-small \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_ontology_train.csv \
  --validation_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/validate_data/QA_ontology_validate.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 20 \
  --learning_rate 5e-4 \
  --num_train_epochs 20 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_sm \
  --overwrite_output_dir \
  --predict_with_generate 
  
python qa.py \
  --model_name_or_path ./outputs/t5_sm \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_ontology_train.csv \
  --test_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/test_data/QA_ontology_test.csv \
  --do_predict \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_sm \
  --overwrite_output_dir \
  --predict_with_generate
  
  ### T5-BS
  
python qa.py \
  --model_name_or_path t5-base \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_ontology_train.csv \
  --validation_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/validate_data/QA_ontology_validate.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 20 \
  --learning_rate 5e-4 \
  --num_train_epochs 20 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_bs \
  --overwrite_output_dir \
  --predict_with_generate 
  
  
python qa.py \
  --model_name_or_path ./outputs/t5_bs \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_ontology_train.csv \
  --test_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/test_data/QA_ontology_test.csv \
  --do_predict \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_bs \
  --overwrite_output_dir \
  --predict_with_generate
  
  
  ### T5-LG
  
python qa.py \
  --model_name_or_path t5-large \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_ontology_train.csv \
  --validation_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/validate_data/QA_ontology_validate.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-4 \
  --num_train_epochs 20 \
  --max_seq_length 64 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_lg \
  --overwrite_output_dir \
  --predict_with_generate 
  
  
python qa.py \
  --model_name_or_path ./outputs/t5_lg \
  --question_column Question \
  --answer_column Answer \
  --context_column Context \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_ontology_train.csv \
  --test_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/test_data/QA_ontology_test.csv \
  --do_predict \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_lg \
  --overwrite_output_dir \
  --predict_with_generate
  
  
  
  ## INCONTEXT
  
### OURS
python qa.py \
  --model_name_or_path t5-small \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_incontext_train.csv \
  --validation_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/validate_data/QA_incontext_validate.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 20 \
  --learning_rate 1e-3 \
  --num_train_epochs 20 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_sm_inctx \
  --overwrite_output_dir \
  --predict_with_generate 
  
python qa.py \
  --model_name_or_path ./outputs/t5_sm_inctx \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_incontext_train.csv \
  --test_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/test_data/QA_incontext_test.csv \
  --do_predict \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_sm_inctx \
  --overwrite_output_dir \
  --predict_with_generate
  
  ### T5-BASE
  
python qa.py \
  --model_name_or_path t5-base \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_incontext_train.csv \
  --validation_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/validate_data/QA_incontext_validate.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 20 \
  --learning_rate 5e-4 \
  --num_train_epochs 20 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_bs_inctx \
  --overwrite_output_dir \
  --predict_with_generate 
  
python qa.py \
  --model_name_or_path ./outputs/t5_bs_inctx \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_incontext_train.csv \
  --test_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/test_data/QA_incontext_test.csv \
  --do_predict \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_bs_inctx \
  --overwrite_output_dir \
  --predict_with_generate
  
  
  
  
  
  ## ZERO PARA
  
### OURS
python qa.py \
  --model_name_or_path t5-small \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_zeropara_train.csv \
  --validation_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/validate_data/QA_zeropara_validate.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 20 \
  --learning_rate 5e-4 \
  --num_train_epochs 20 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_sm_zero \
  --overwrite_output_dir \
  --predict_with_generate 
  
python qa.py \
  --model_name_or_path ./outputs/t5_sm_zero \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_zeropara_train.csv \
  --test_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/test_data/QA_zeropara_test.csv \
  --do_predict \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_sm_zero \
  --overwrite_output_dir \
  --predict_with_generate
  
  ### T5-BASE
  
python qa.py \
  --model_name_or_path t5-base \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_zeropara_train.csv \
  --validation_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/validate_data/QA_zeropara_validate.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 20 \
  --learning_rate 1e-4 \
  --num_train_epochs 20 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_bs_zero \
  --overwrite_output_dir \
  --predict_with_generate 
  
python qa.py \
  --model_name_or_path ./outputs/t5_bs_zero \
  --question_column Question \
  --answer_column Answer \
  --context_column Para \
  --train_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/train_data/QA_zeropara_train.csv \
  --test_file /home/kkpal/AISECKG-QA-Dataset/Chat-GPT/Jsonfiles/test_data/QA_zeropara_test.csv \
  --do_predict \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_steps 1000 \
  --output_dir ./outputs/t5_bs_zero \
  --overwrite_output_dir \
  --predict_with_generate
