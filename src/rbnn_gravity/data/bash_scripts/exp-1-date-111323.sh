CUDA_VISIBLE_DEVICE=0 python src/rbnn_gravity/data/generate_data.py --name='exp-1'\
                                                                --date='111323'\
                                                                --save_dir='src/rbnn_gravity/data/generated_datasets'\
                                                                --n_examples=1000\
                                                                --traj_len=100\
                                                                --dt=0.001\
                                                                --mass=1.\
                                                                --radius=50.\
                                                                --ic_type='random'\
                                                                --seed=0\
                                                                --moi_diag_gt 3. 2. 1.\
                                                                --moi_off_diag_gt 0. 0. 0.\
                                                