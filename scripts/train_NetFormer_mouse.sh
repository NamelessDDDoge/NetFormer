python train_NetFormer_mouse.py \
    --out_folder='../output/NetFormer_mouse/' \
    --input_mouse='SB025' \
    --input_sessions='2019-10-23' \
    --window_size=60 \
    --predict_window_size=1 \
    --learning_rate=1e-3 \
    --scheduler=plateau \
    --dim_E=30 \
    --model_random_seed=4