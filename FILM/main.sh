rm -rf pictures/
python main.py  -n1 --max_episode_length 1000 --num_local_steps 25  --num_processes 1 --eval_split tests_unseen --from_idx 0 --to_idx 120 --max_fails 10   --debug_local  --use_sem_seg --set_dn first_run   --use_sem_policy  --save_pictures -v 1 --which_gpu 0 --x_display 1 --dialfred
