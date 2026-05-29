python train_LagFormer_mouse.py \
    --batch_size=8 \
    --out_folder='../output/LagFormer_mouse/' \
    --input_mouse='SB025' \
    --input_sessions='2019-10-23' \
    --window_size=60 \
    --predict_window_size=1 \
    --learning_rate=1e-3 \
    --scheduler=plateau \
    --dim_E=30 \