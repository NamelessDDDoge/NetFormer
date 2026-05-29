import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import r2_score

from NetFormer import data, tools, lagFormer


if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")

    parser.add_argument("--out_folder", help="the output folder")

    # Data

    parser.add_argument("--input_mouse")
    parser.add_argument("--input_sessions")

    parser.add_argument("--window_size", default=200)
    parser.add_argument("--predict_window_size", default=1)
    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)

    # Model
    
    parser.add_argument("--model_random_seed", default=42)
    parser.add_argument("--learning_rate", help="learning rate", default=1e-4)

    parser.add_argument("--attention_activation", default="none")    # "softmax" or "sigmoid" or "tanh" or "none"
    parser.add_argument("--scheduler", default="plateau")
    parser.add_argument("--weight_decay", default=0)

    parser.add_argument("--dim_E", default=100)
    parser.add_argument("--constraint_loss_weight", default=0)

    args = parser.parse_args()

    # Set the hyperparameters
    out_folder = args.out_folder

    # Data

    input_mouse = [str(mouse) for mouse in args.input_mouse.split('|')]
    input_sessions = [str(mouse) for mouse in args.input_sessions.split('|')]
    for i in range(len(input_sessions)):
        input_sessions[i] = input_sessions[i].split('_')

    window_size = int(args.window_size)
    predict_window_size = int(args.predict_window_size)
    batch_size = int(args.batch_size)

    # Model

    model_random_seed = int(args.model_random_seed)
    learning_rate = float(args.learning_rate)

    attention_activation = args.attention_activation
    scheduler = args.scheduler
    weight_decay = float(args.weight_decay)

    dim_E = int(args.dim_E)
    constraint_loss_weight = float(args.constraint_loss_weight)


    output_path = (
        out_folder
        + args.input_mouse
        + "_"
        + args.input_sessions
        + "_"
        + str(window_size)
        + "_"
        + str(predict_window_size)
        + "_"
        + str(batch_size)
        + "_"
        + str(model_random_seed)
        + "_"
        + str(learning_rate)
        + "_"
        + attention_activation
        + "_"
        + scheduler
        + "_"
        + str(weight_decay)
        + "_"
        + str(dim_E)
        + "_"
        + str(constraint_loss_weight)
    )

    checkpoint_path = output_path
    log_path = out_folder + "/log"


    train_dataloader, val_dataloader, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, num_batch_per_session_TRAIN, num_batch_per_session_VAL, sessions_2_original_cell_type, neuron_id_2_cell_type_id = data.generate_mouse_all_sessions_data(
        input_mouse=input_mouse,
        input_sessions=input_sessions,
        window_size=window_size,
        batch_size=batch_size,
    )

    single_model = lagFormer.LagFormerMouse(
        model_random_seed=model_random_seed,
        attention_activation=attention_activation,
        dim_E=dim_E
    )

    es = EarlyStopping(monitor="VAL_loss", patience=20)
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="VAL_loss", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor()
    logger = TensorBoardLogger(log_path, name="model")
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
        benchmark=False,
        profiler="simple",
        logger=logger,
        max_epochs=100,
        gradient_clip_val=0,
    )

    trainer.fit(single_model, train_dataloader, val_dataloader)


    ############################################################################################################
    # Evaluate Model Performance
    ############################################################################################################

    eval_cell_type_order = ['EC', 'Pvalb', 'Sst', 'Vip']
    GT_strength_connectivity = np.zeros((len(eval_cell_type_order), len(eval_cell_type_order)))
    GT_strength_connectivity[:] = np.nan

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('EC')] = 0.11
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('EC')] = 0.27
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('EC')]= 0.1
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('EC')] = 0.45

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Pvalb')] = -0.44
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Pvalb')] = -0.47
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Pvalb')] = -0.44
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Pvalb')] = -0.23

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Sst')] = -0.16
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Sst')] = -0.18
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Sst')] = -0.19
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Sst')] = -0.17

    GT_strength_connectivity[eval_cell_type_order.index('EC')][eval_cell_type_order.index('Vip')] = -0.06
    GT_strength_connectivity[eval_cell_type_order.index('Pvalb')][eval_cell_type_order.index('Vip')] = -0.10
    GT_strength_connectivity[eval_cell_type_order.index('Sst')][eval_cell_type_order.index('Vip')] = -0.17
    GT_strength_connectivity[eval_cell_type_order.index('Vip')][eval_cell_type_order.index('Vip')] = -0.10

    max_abs = np.max(np.abs(GT_strength_connectivity))
    vmin_KK = -max_abs
    vmax_KK = max_abs


    model_checkpoint_path = checkpoint_callback.best_model_path
    if not model_checkpoint_path:
        raise RuntimeError("No model checkpoint was saved during training.")

    train_results = trainer.predict(single_model, dataloaders=[train_dataloader], ckpt_path=model_checkpoint_path)

    attentions = []  # list of (N * neuron_num * neuron_num)
    attentions_by_state = []  # list of (3 * N * neuron_num * neuron_num)
    all_sessions_avg_attention_NN = []  # list of (neuron_num * neuron_num)

    index = 0
    num_session = len(num_batch_per_session_TRAIN)
    for i in range(num_session):
        attentions.append([])
        # attentions_by_state.append([[], [], []]) # 3 states for each session

        for j in range(num_batch_per_session_TRAIN[i]):
            x_hat = train_results[index][0]
            x = train_results[index][1]
            attention = train_results[index][2]  # B * neuron_num * neuron_num
            state = train_results[index][3].cpu().numpy()   # B * window_size

            attentions[i].append(attention)

            index += 1

        attentions[i] = torch.cat(attentions[i], dim=0).cpu().numpy()    # N * neuron_num * neuron_num
        # get average attention across samples in each session
        all_sessions_avg_attention_NN.append(np.mean(attentions[i], axis=0))   # neuron_num * neuron_num


    # Validation Result

    val_results = trainer.predict(single_model, dataloaders=[val_dataloader], ckpt_path=model_checkpoint_path)

    predictions = []
    ground_truths = []

    index = 0
    num_session = len(num_batch_per_session_VAL)
    for i in range(num_session):
        predictions.append([])
        ground_truths.append([])

        for j in range(num_batch_per_session_VAL[i]):
            x_hat = val_results[index][0]
            x = val_results[index][1]
            
            predictions[i].append(x_hat)
            ground_truths[i].append(x)
            index += 1
        
        predictions[i] = torch.cat(predictions[i], dim=0).cpu().numpy()  # N * neuron_num * window_size
        ground_truths[i] = torch.cat(ground_truths[i], dim=0).cpu().numpy()  # N * neuron_num * window_size

    flatten_predictions = [predictions[0].flatten()]
    flatten_ground_truths = [ground_truths[0].flatten()]
    for i in range(1, num_session):
        flatten_predictions.append(predictions[i].flatten())
        flatten_ground_truths.append(ground_truths[i].flatten())

    flatten_predictions = np.concatenate(flatten_predictions)
    flatten_ground_truths = np.concatenate(flatten_ground_truths)

    pred_corr = stats.pearsonr(flatten_predictions, flatten_ground_truths)[0]
    R_squared = r2_score(flatten_ground_truths, flatten_predictions)
    mse = np.mean((flatten_predictions - flatten_ground_truths) ** 2)

    plt.scatter(flatten_predictions, flatten_ground_truths, s=1)
    plt.title("val_corr = " + str(pred_corr)[:7] + ", R^2 = " + str(R_squared)[:7] + ", mse = " + str(mse)[:7])
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.savefig(output_path + "/pred.png")
    plt.close()

    # Evaluate Attention Maps

    multisession_NN_list = all_sessions_avg_attention_NN
    experiment_KK_strength = tools.multisession_NN_to_KK(
        multisession_NN_list, 
        cell_type_order,
        all_sessions_new_cell_type_id,
    )

    eval_cell_type_order = ['EC', 'Pvalb', 'Sst', 'Vip']
    eval_KK_strength = tools.experiment_KK_to_eval_KK(experiment_KK_strength, cell_type_order, eval_cell_type_order)

    pearson_corr_strength_KK = stats.pearsonr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]
    spearman_corr_strength_KK = stats.spearmanr(GT_strength_connectivity.flatten(), eval_KK_strength.flatten())[0]

    plt.imshow(tools.linear_transform(eval_KK_strength, GT_strength_connectivity), cmap='RdBu_r', interpolation="nearest", vmin=vmin_KK, vmax=vmax_KK)
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("KK strength, pearson = " + str(pearson_corr_strength_KK)[:7] + ", spearman = " + str(spearman_corr_strength_KK)[:7])
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/strength.png")
    plt.close()

    np.save(output_path + "/Estimated_strength.npy", eval_KK_strength)

    plt.imshow(GT_strength_connectivity, cmap='RdBu_r', interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Pre")
    plt.ylabel("Post")
    plt.title("GT strength connectivity")
    plt.xticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order, rotation=45)
    plt.yticks(np.arange(len(eval_cell_type_order)), eval_cell_type_order)
    plt.savefig(output_path + "/GT_strength.png")
    plt.close()
