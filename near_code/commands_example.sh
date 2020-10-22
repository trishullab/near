python train.py \
--algorithm mcts \
--exp_name example \
--trial 1 \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 5 \
--num_mc_samples 5

python train.py \
--algorithm astar-near \
--exp_name example \
--trial 1 \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 5 \
--frontier_capacity 5

python train.py \
--algorithm iddfs-near \
--exp_name example \
--trial 1 \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 5 \
--frontier_capacity 5 \
--initial_depth 3

python train.py \
--algorithm mc-sampling \
--exp_name example \
--trial 1 \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 5 \
--num_mc_samples 5

python train.py \
--algorithm enumeration \
--exp_name example \
--trial 1 \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 5 \
--max_num_programs 10

python train.py \
--algorithm genetic \
--exp_name example \
--trial 1 \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 5 \
--population_size 10 \
--selection_size 5 \
--num_gens 2 \
--total_eval 20 \
--mutation_prob 0.1 \
--max_enum_depth 5

python train.py \
--algorithm rnn \
--exp_name example \
--trial 1 \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--lossfxn "crossentropy" \
--normalize \
--neural_epochs 10 \
--learning_rate 0.01 \
--max_num_units 64

python eval.py \
--program_path results/example_astar-near_001/program.p \
--train_data data/example/train_ex_data.npy \
--test_data data/example/test_ex_data.npy \
--train_labels data/example/train_ex_labels.npy \
--test_labels data/example/test_ex_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 12 \
--output_size 4 \
--num_labels 4 \
--normalize 
