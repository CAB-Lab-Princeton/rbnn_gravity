CUDA_VISIBLE_DEVICE=0 python src/rbnn_gravity/run.py --exp_name='test_run'\
                                    --date='11122023'\
                                    --data_dir='src/rbnn_gravity/data/test_folder/'\
                                    --save_dir='src/rbnn_gravity/test_models/'\
                                    --gpu_id=0\
                                    --seed=0\
                                    --n_epochs=1000\
                                    --seq_len=2\
                                    --batch_size=16\
                                    --lr=1e-3\
                                    --dt=1e-3\
                                    --tau=2\
                                    --V_in_dim=9\
                                    --V_hidden_dim=15\
                                    --V_out_dim=1\
                                    --lambda_loss 1. 1.\
                                    --print_every=1\
                                    --retrain_model\
                                    --save_model\

