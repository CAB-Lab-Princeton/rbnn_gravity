CUDA_VISIBLE_DEVICE=0 python src/rbnn_gravity/data/generate_data.py --name='exp-uniform'\
                                                                --date='120323'\
                                                                --save_dir='src/rbnn_gravity/data/generated_datasets'\
                                                                --n_examples=1000\
                                                                --traj_len=100\
                                                                --dt=0.001\
                                                                --mass=1.\
                                                                --radius=50.\
                                                                --R_ic_type='uniform'\
                                                                --pi_ic_type='random'\
                                                                --seed=0\
                                                                --moi_diag_gt 1. 2.8 2.\
                                                                --moi_off_diag_gt 0. 0. 0.\
                                                