import torch
import torch.nn.functional as F
from torch.nn.modules import Module
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from os.path import dirname, join as pjoin
import pandas as pd
from NetFormer import tools


####################################################################################################
# Connectivity-constrained simulation
####################################################################################################

class data_simulator(Module):
    """
    data simulator for neuron activity
    Formula:
        x(t+1) = tanh( W @ x(t+1-tau) + b ) + noise(t)
    """

    def __init__(
        self, 
        neuron_num: int, 
        tau=1,    # integer > 0
        weight_scale=1,
        init_scale=1,
        error_scale=3.5,
        total_time=30000,
        data_random_seed=42,
    ):
        super().__init__()
        self.error_scale = error_scale

        self.neuron_num = neuron_num
        self.tau = tau

        torch.manual_seed(data_random_seed)
        np.random.seed(data_random_seed)

        self.b = weight_scale * torch.randn(neuron_num)    # constant input for each neuron

        self.activity = []     # store the activity of each neuron at each time step so far（total_time, neuron_num)
        for i in range(int(tau)):
            self.activity.append(init_scale * torch.randn(neuron_num))

        self.W_ij, self.cell_type_order, self.cell_type_ids, self.cell_type_count = tools.construct_weight_matrix_cell_type(neuron_num)
        self.W_ij = weight_scale * self.W_ij

    def forward(self, current_time_step):
        signal = F.tanh((self.W_ij @ self.activity[-int(self.tau)]) + self.b)
        e = torch.normal(0,1,size=(self.neuron_num,)) * self.error_scale
        x_t_1 = signal + e

        self.activity.append(x_t_1)
        return x_t_1   # a vector of size neuron_num
    


def generate_simulation_data(
    neuron_num=200,
    tau=1,
    weight_scale=1,
    init_scale=1,
    error_scale=3.5,
    total_time=30000,
    data_random_seed=42,
    window_size=200,
    predict_window_size=1,
    batch_size=32,
    num_workers: int=6, 
    split_ratio=0.8,
    model_type="NetFormer",    # "NetFormer" or "RNN"
    data_type="connectivity_constrained",    # "connectivity_constrained" or "ring_circuit"
    spatial_partial_measurement=200,  # the number of neurons that is measured, between 0 and neuron_num
) -> DataLoader:
    """
    Generate dataset.

    Returns
    -------
    dataloaders and ground truth weight matrix.
    """

    if data_type == "connectivity_constrained":

        simulator = data_simulator(
            neuron_num=neuron_num, 
            tau=tau,  
            weight_scale=weight_scale,
            init_scale=init_scale,
            error_scale=error_scale,
            total_time=total_time,
            data_random_seed=data_random_seed,
        )

        data = []
        for t in range(total_time):
            x_t = simulator.forward(t)
            x_t = x_t.view(-1, 1)
            data.append(x_t)
        data = torch.cat(data, dim=1).float()

    elif data_type == "ring_circuit":

        np.random.seed(data_random_seed)
        W = tools.construct_weight_matrix_ring_circuit(neuron_num)

        activity = []     # store the activity of each neuron at each time step so far（total_time x neuron_num)
        for i in range(1):
            activity.append(np.zeros(neuron_num))

        for t in range(total_time):
            noise_sd = 1
            noise_sparsity = 1
            b = 1
            r = weight_scale

            noise = noise_sd * np.random.randn(neuron_num) * (np.random.randn(neuron_num) > noise_sparsity)
            signal = np.tanh(r * (W @ activity[-1]) + b * (1+noise))
            x_t_1 = signal
            activity.append(x_t_1)

        activity = np.array(activity)
        # remove the first row
        activity = activity[1:]
        activity = activity.T
        data = torch.from_numpy(activity).float()

        total_time = data.shape[1]
        neuron_num = data.shape[0]

    else:
        raise ValueError("data_type should be either 'connectivity_constrained' or 'ring_circuit'.")

    # Partial Observation

    if spatial_partial_measurement != neuron_num:
        # choose a subset of neurons to measure, without replacement
        idx = torch.randperm(neuron_num)[:spatial_partial_measurement]
        # sort the indices
        idx = torch.sort(idx)[0]
        data = data[idx, :]

    # First split_ratio of the data is for training (total_time * split_ratio), the rest is for validation

    train_data_length = int(total_time * split_ratio)
    train_data = data[:, :train_data_length]
    val_data = data[:, train_data_length:]

    if (model_type == "NetFormer"):

        val_data_size = val_data.shape[1] - window_size + 1
        val_start_indices = torch.arange(val_data_size)

        val_data_result = []
        for i in range(val_data_size):
            index = val_start_indices[i]
            sample = val_data[:, index:index+window_size]
            if spatial_partial_measurement != neuron_num:
                val_data_result.append(sample.view(1, spatial_partial_measurement, window_size))
            else:
                val_data_result.append(sample.view(1, neuron_num, window_size))
        val_data = torch.cat(val_data_result, dim=0)

        train_data_size = train_data_length - window_size + 1
        train_start_indices = torch.arange(train_data_size)

        train_data_result = []
        for i in range(train_data_size):
            index = train_start_indices[i]
            sample = train_data[:, index:index+window_size]
            if spatial_partial_measurement != neuron_num:
                train_data_result.append(sample.view(1, spatial_partial_measurement, window_size))
            else:
                train_data_result.append(sample.view(1, neuron_num, window_size))
        train_data = torch.cat(train_data_result, dim=0)

        train_dataset = TensorDataset(train_data[:, :, :-predict_window_size], train_data[:, :, -predict_window_size:])
        val_dataset = TensorDataset(val_data[:, :, :-predict_window_size], val_data[:, :, -predict_window_size:])

    elif (model_type == "RNN"):

        # input takes in activity from one previous time step to predict for the next time step
        train_x = train_data[:, :-1].transpose(0, 1)
        train_y = train_data[:, 1:].transpose(0, 1)
        val_x = val_data[:, :-1].transpose(0, 1)
        val_y = val_data[:, 1:].transpose(0, 1)

        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)

    else:
        raise ValueError("model_type should be either 'NetFormer' or 'RNN'.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    if data_type == "connectivity_constrained":

        if spatial_partial_measurement == neuron_num:
            return train_dataloader, val_dataloader, simulator.W_ij, simulator.cell_type_ids, simulator.cell_type_order, simulator.cell_type_count
        else:
            new_W_ij = torch.zeros((spatial_partial_measurement, spatial_partial_measurement))
            for i in range(spatial_partial_measurement):
                for j in range(spatial_partial_measurement):
                    new_W_ij[i][j] = simulator.W_ij[idx[i]][idx[j]]

            new_cell_type_ids = []
            for i in range(spatial_partial_measurement):
                new_cell_type_ids.append(simulator.cell_type_ids[idx[i]])

            id2cell_type = {0:'EC', 1:'Pvalb', 2:'Sst', 3:'Vip'}
            new_cell_type_count = {'EC':0, 'Pvalb':0, 'Sst':0, 'Vip':0}
            for i in range(spatial_partial_measurement):
                cell_type = id2cell_type[new_cell_type_ids[i]]
                new_cell_type_count[cell_type] += 1

            return train_dataloader, val_dataloader, new_W_ij, new_cell_type_ids, simulator.cell_type_order, new_cell_type_count
        
    elif data_type == "ring_circuit":

        if spatial_partial_measurement == neuron_num:
            return train_dataloader, val_dataloader, W
        else:
            raise ValueError("For ring circuit, spatial_partial_measurement is not supported.")
    



############################################################################################################
# Multimodal mouse data with multiple sessions
############################################################################################################

def load_mouse_data_session(directory, date_exp, input_setting):
    gene_count = np.load(directory + date_exp + 'neuron.gene_count.npy')
    UniqueID = np.load(directory + date_exp + 'neuron.UniqueID.npy')

    with open(directory + date_exp + 'neuron.ttype.txt') as f:
        neuron_ttypes_raw  = f.readlines()
    neuron_ttypes = []
    for neuron_ttype in neuron_ttypes_raw:
        neuron_ttypes.append(neuron_ttype.strip())

    frame_states = np.load(directory + date_exp + input_setting + 'frame.states.npy')
    frame_times = np.load(directory + date_exp + input_setting + 'frame.times.npy')
    frame_activity = np.load(directory + date_exp + input_setting + 'frame.neuralActivity.npy')

    activity_norm = frame_activity

    return np.transpose(activity_norm), frame_states, frame_times, UniqueID.reshape(-1), neuron_ttypes



class Mouse_All_Sessions_Dataset(TensorDataset):
    """
    Define a dataset to handle the issue of different number of neurons in different sessions.
    """

    def __init__(
        self, 
        all_sessions_activity_windows,  # list of 3d tensors, each tensor is a session (num_window, n, window_size)) 
        all_sessions_new_UniqueID_windows,  # list of 2d tensors, each tensor is a session (num_window, n)
        all_sessions_new_cell_type_id_windows, # list of 2d tensors, each tensor is a session (num_window, n)
        all_sessions_state_windows,  # list of 2d tensors, each tensor is a session (num_window, window_size)
        batch_size=3,                      # real batch size
    ):
        self.num_batch_per_session = [session.shape[0] // batch_size for session in all_sessions_activity_windows]

        self.all_batch = []
        self.all_batch_neuron_ids = []
        self.all_batch_cell_type_ids = []
        self.all_batch_states = []
        for i in range(len(self.num_batch_per_session)):      # for each session
            for j in range(self.num_batch_per_session[i]):      # for each batch
                self.all_batch.append(torch.Tensor(all_sessions_activity_windows[i][j*batch_size:(j+1)*batch_size]).float())
                self.all_batch_neuron_ids.append(torch.Tensor(all_sessions_new_UniqueID_windows[i][j*batch_size:(j+1)*batch_size]).int())
                self.all_batch_cell_type_ids.append(torch.Tensor(all_sessions_new_cell_type_id_windows[i][j*batch_size:(j+1)*batch_size]).int())
                self.all_batch_states.append(torch.Tensor(all_sessions_state_windows[i][j*batch_size:(j+1)*batch_size]).int())

    def __getitem__(self, index):
        return self.all_batch[index], self.all_batch_neuron_ids[index], self.all_batch_cell_type_ids[index], self.all_batch_states[index]

    def __len__(self):
        return len(self.all_batch)
    


def generate_mouse_all_sessions_data(
    input_mouse: list,    # e.g. [SB025, SB026]
    input_sessions: list,    # e.g. [[2019-10-07, 2019-10-04], [2019-10-11, 2019-10-14, 2019-10-16]]
    window_size=200, 
    batch_size=32,
    num_workers: int=6, 
    split_ratio=0.8,
):
    
    directory = '../data/Mouse/Bugeon/'
    input_sessions_file_path = []
    for i in range(len(input_mouse)):
        for j in range(len(input_sessions[i])):
            input_sessions_file_path.append({'date_exp': input_mouse[i] + '/' + input_sessions[i][j] + '/', 'input_setting': 'Blank/01/'})
            print(input_mouse[i] + '/' + input_sessions[i][j])

    all_sessions_original_UniqueID = []
    all_sessions_original_cell_type = []

    all_sessions_acitvity_TRAIN = []
    all_sessions_acitvity_VAL = []

    all_sessions_state_TRAIN = []
    all_sessions_state_VAL = []

    num_neurons_per_session = []
    sessions_2_original_cell_type = []
    all_sessions_activity_flatten = []

    for i in range(len(input_sessions_file_path)):
        date_exp = input_sessions_file_path[i]['date_exp']
        input_setting = input_sessions_file_path[i]['input_setting']
        activity, frame_states, frame_times, UniqueID, neuron_ttypes = load_mouse_data_session(
            directory, date_exp, input_setting,
        )
        frame_states = frame_states.flatten()

        all_sessions_original_UniqueID.append(UniqueID)
        all_sessions_acitvity_TRAIN.append(activity[:, :int(activity.shape[1]*split_ratio)])
        all_sessions_acitvity_VAL.append(activity[:, int(activity.shape[1]*split_ratio):])
        all_sessions_state_TRAIN.append(frame_states[:int(activity.shape[1]*split_ratio)])
        all_sessions_state_VAL.append(frame_states[int(activity.shape[1]*split_ratio):])
        num_neurons_per_session.append(activity.shape[0])
        all_sessions_activity_flatten.append(activity.flatten())

        # get the first level of cell types
        neuron_types_result = []
        for j in range(len(neuron_ttypes)):
            # split by "-"
            neuron_types_result.append(neuron_ttypes[j].split("-")[0])

        sessions_2_original_cell_type.append(neuron_types_result)
        all_sessions_original_cell_type.append(neuron_types_result)

    all_sessions_original_UniqueID = np.concatenate(all_sessions_original_UniqueID)
    all_sessions_original_cell_type = np.concatenate(all_sessions_original_cell_type)
    all_sessions_activity_flatten = np.concatenate(all_sessions_activity_flatten)
    mu = np.mean(all_sessions_activity_flatten)
    std = np.std(all_sessions_activity_flatten)

    # all_sessions normalization
    all_sessions_acitvity_TRAIN = [(session - mu) / std for session in all_sessions_acitvity_TRAIN]
    all_sessions_acitvity_VAL = [(session - mu) / std for session in all_sessions_acitvity_VAL]
    
    ##############################################
    # Construct new UniqueID and cell type id
    ##############################################

    all_sessions_new_UniqueID, num_unqiue_neurons = tools.assign_unique_neuron_ids(all_sessions_original_UniqueID, num_neurons_per_session)
    all_sessions_new_cell_type_id, cell_type_order = tools.assign_unique_cell_type_ids(all_sessions_original_cell_type, num_neurons_per_session)

    neuron_id_2_cell_type_id = np.zeros((num_unqiue_neurons,)).astype(int)
    for i in range(len(all_sessions_new_UniqueID)):
        neuron_id_2_cell_type_id[all_sessions_new_UniqueID[i].astype(int)] = all_sessions_new_cell_type_id[i]

    ##############################################
    # Construct windows
    ##############################################

    # For TRAIN
    all_sessions_activity_windows_TRAIN, all_sessions_new_UniqueID_windows_TRAIN, all_sessions_new_cell_type_id_window_TRAIN, all_sessions_state_windows_TRAIN = tools.sliding_windows(
        all_sessions_acitvity_TRAIN, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, all_sessions_state_TRAIN, window_size=window_size
    )
    # For VAL
    all_sessions_activity_windows_VAL, all_sessions_new_UniqueID_windows_VAL, all_sessions_new_cell_type_id_window_VAL, all_sessions_state_windows_VAL = tools.sliding_windows(
        all_sessions_acitvity_VAL, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, all_sessions_state_VAL, window_size=window_size
    )

    ##############################################
    # Construct dataloaders
    ##############################################

    train_dataset = Mouse_All_Sessions_Dataset(
        all_sessions_activity_windows_TRAIN, 
        all_sessions_new_UniqueID_windows_TRAIN, 
        all_sessions_new_cell_type_id_window_TRAIN, 
        all_sessions_state_windows_TRAIN,
        batch_size=batch_size,        # real batch_size
    )
    val_dataset = Mouse_All_Sessions_Dataset(
        all_sessions_activity_windows_VAL, 
        all_sessions_new_UniqueID_windows_VAL, 
        all_sessions_new_cell_type_id_window_VAL, 
        all_sessions_state_windows_VAL,
        batch_size=batch_size,        # real batch_size
    )

    num_batch_per_session_TRAIN = train_dataset.num_batch_per_session
    num_batch_per_session_VAL = val_dataset.num_batch_per_session

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True)    # 1 is not real batch_size
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True)        # 1 is not real batch_size

    return train_dataloader, val_dataloader, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, num_batch_per_session_TRAIN, num_batch_per_session_VAL, sessions_2_original_cell_type, neuron_id_2_cell_type_id



def generate_mouse_all_sessions_data_for_RNN(
    input_mouse: list,    # e.g. [SB025, SB026]
    input_sessions: list,    # e.g. [[2019-10-07, 2019-10-04], [2019-10-11, 2019-10-14, 2019-10-16]]
    k=1,     # the number of tau(s) to consider, each tau corresponds to one linear trnasformation matrix A in RNN
    batch_size=32,
    num_workers: int=6, 
    normalization = 'all',    # destd for RNN with exp, all for RNN with tanh
    split_ratio=0.8,
):
    """
    Mouse data generation for RNN models. It should return a list of dataloaders for each session.
    RNN is trained on one session at a time. x_(t+1) = \sum A_k x_(t-k)
    The input is previous K time steps and the output is the current ONE time step.
    """
    
    directory = '../data/Mouse/Bugeon/'
    input_sessions_file_path = []
    for i in range(len(input_mouse)):
        for j in range(len(input_sessions[i])):
            input_sessions_file_path.append({'date_exp': input_mouse[i] + '/' + input_sessions[i][j] + '/', 'input_setting': 'Blank/01/'})

    all_sessions_original_UniqueID = []
    all_sessions_original_cell_type = []

    all_sessions_acitvity_TRAIN = []
    all_sessions_acitvity_VAL = []

    all_sessions_state_TRAIN = []
    all_sessions_state_VAL = []

    num_neurons_per_session = []
    sessions_2_original_cell_type = []
    all_sessions_activity_flatten = []

    for i in range(len(input_sessions_file_path)):
        date_exp = input_sessions_file_path[i]['date_exp']
        input_setting = input_sessions_file_path[i]['input_setting']

        activity, frame_states, frame_times, UniqueID, neuron_ttypes = load_mouse_data_session(
            directory, date_exp, input_setting,
        )
        frame_states = frame_states.flatten()

        all_sessions_original_UniqueID.append(UniqueID)
        all_sessions_acitvity_TRAIN.append(activity[:, :int(activity.shape[1]*split_ratio)])
        all_sessions_acitvity_VAL.append(activity[:, int(activity.shape[1]*split_ratio):])
        all_sessions_state_TRAIN.append(frame_states[:int(activity.shape[1]*split_ratio)])
        all_sessions_state_VAL.append(frame_states[int(activity.shape[1]*split_ratio):])
        num_neurons_per_session.append(activity.shape[0])

        all_sessions_activity_flatten.append(activity.flatten())

        # Get the first level of cell types
        neuron_types_result = []
        for j in range(len(neuron_ttypes)):
            # split by "-"
            neuron_types_result.append(neuron_ttypes[j].split("-")[0])

        sessions_2_original_cell_type.append(neuron_types_result)
        all_sessions_original_cell_type.append(neuron_types_result)

    all_sessions_original_UniqueID = np.concatenate(all_sessions_original_UniqueID)     # flatten to 1d
    all_sessions_original_cell_type = np.concatenate(all_sessions_original_cell_type)   # flatten to 1d

    all_sessions_activity_flatten = np.concatenate(all_sessions_activity_flatten)
    mu = np.mean(all_sessions_activity_flatten)
    std = np.std(all_sessions_activity_flatten)

    if normalization == 'all':
        all_sessions_acitvity_TRAIN = [(session - mu) / std for session in all_sessions_acitvity_TRAIN]
        all_sessions_acitvity_VAL = [(session - mu) / std for session in all_sessions_acitvity_VAL]
    elif normalization == 'destd':
        all_sessions_acitvity_TRAIN = [(session) / std for session in all_sessions_acitvity_TRAIN]
        all_sessions_acitvity_VAL = [(session) / std for session in all_sessions_acitvity_VAL]


    ##############################################
    # Construct new UniqueID and cell type id
    ##############################################

    # all_essions_new_UniqueID: (num_sessions, num_neurons_per_session), num_unqiue_neurons: a number, total number of unique neurons
    all_sessions_new_UniqueID, num_unqiue_neurons = tools.assign_unique_neuron_ids(all_sessions_original_UniqueID, num_neurons_per_session)
    # all_sessions_new_cell_type_id: (num_sessions, num_neurons_per_session), cell_type_order: a list of cell types, the index corresponds to cell type id
    all_sessions_new_cell_type_id, cell_type_order = tools.assign_unique_cell_type_ids(all_sessions_original_cell_type, num_neurons_per_session)

    neuron_id_2_cell_type_id = np.zeros((num_unqiue_neurons,)).astype(int)
    for i in range(len(all_sessions_new_UniqueID)):
        neuron_id_2_cell_type_id[all_sessions_new_UniqueID[i].astype(int)] = all_sessions_new_cell_type_id[i]


    ##############################################
    # Construct windows
    ##############################################

    # For TRAIN
    # all_sessions_activity_windows: a list of sessions activity windows, each session is a 3D array of shape num_windows x num_neurons x window_size
    all_sessions_activity_windows_TRAIN, all_sessions_new_UniqueID_windows_TRAIN, all_sessions_new_cell_type_id_window_TRAIN, all_sessions_state_windows_TRAIN = tools.sliding_windows(
        all_sessions_acitvity_TRAIN, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, all_sessions_state_TRAIN, window_size=(k+1)
    )
    # For VAL
    all_sessions_activity_windows_VAL, all_sessions_new_UniqueID_windows_VAL, all_sessions_new_cell_type_id_window_VAL, all_sessions_state_windows_VAL = tools.sliding_windows(
        all_sessions_acitvity_VAL, all_sessions_new_UniqueID, all_sessions_new_cell_type_id, all_sessions_state_VAL, window_size=(k+1)
    )

    ##############################################
    # Construct dataloaders
    ##############################################

    train_dataloader_list = []
    val_dataloader_list = []

    for i in range(len(input_sessions_file_path)):
        train_data = torch.Tensor(all_sessions_activity_windows_TRAIN[i]).float()
        val_data = torch.Tensor(all_sessions_activity_windows_VAL[i]).float()

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

    return train_dataloader_list, val_dataloader_list, num_unqiue_neurons, cell_type_order, all_sessions_new_cell_type_id, sessions_2_original_cell_type, neuron_id_2_cell_type_id