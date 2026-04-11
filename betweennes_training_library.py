import torch
import numpy as np
from scipy.stats import kendalltau

def sparse_mx_to_torch_sparse_tensor(sparse_mx, device):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)

def train(adjacency_matrices, adjacency_matrices_T, nodes_list, betweenness, model, device, optimizer, model_size, writer, epoch, batch_size=16):
    model.train()
    loss_total = 0
    num_samples = len(adjacency_matrices)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # Per mescolare i dati

    for batch_start in range(0, num_samples, batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]

        optimizer.zero_grad()
        batch_loss = 0

        for i in batch_indices:
            adj = sparse_mx_to_torch_sparse_tensor(adjacency_matrices[i], device)
            adj_t = sparse_mx_to_torch_sparse_tensor(adjacency_matrices_T[i], device)
            num_nodes = nodes_list[i]

            y_out = model(adj, adj_t)
            true_arr = torch.from_numpy(np.array(list(betweenness[i].values()))).float().to(device)

            loss_rank = loss_cal(y_out, true_arr, num_nodes, device, model_size)
            batch_loss += loss_rank

        # Backpropagation su tutto il batch
        batch_loss /= len(batch_indices)  # Media delle loss
        batch_loss.backward()
        optimizer.step()

        loss_total += float(batch_loss)

    avg_loss = loss_total / (num_samples // batch_size)
    writer.add_scalar("Loss/train", avg_loss, epoch)

    return avg_loss


def test(adjacency_matrices, adiacency_matrices_T, nodes_list, betweenness, model, device, optimizer, model_size, writer, epoch):
    model.eval()
    loss = 0
    list_kt = list()
    num_samples = len(adjacency_matrices)
    for j in range(num_samples):
        adj = sparse_mx_to_torch_sparse_tensor(adjacency_matrices[j], device)
        adj_t = sparse_mx_to_torch_sparse_tensor(adiacency_matrices_T[j], device)
        num_nodes = nodes_list[j]

        y_out = model(adj, adj_t)

        true_arr = torch.from_numpy(np.array(list(betweenness[j].values()))).float()
        true_val = true_arr.to(device)

        loss_rank = loss_cal(y_out, true_arr, num_nodes, device, model_size)
        loss += float(loss_rank)

        kt = ranking_correlation(y_out, true_val, num_nodes, model_size)
        list_kt.append(kt)

    avg_loss = loss / num_samples
    writer.add_scalar("Loss/test", avg_loss, epoch)
    writer.add_scalar("Average KT score",np.mean(np.array(list_kt)), epoch)
    writer.add_scalar("Average KT standard deviation",np.std(np.array(list_kt)), epoch)

    return avg_loss

def ranking_correlation(y_out, true_val, node_num, model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()

    kt, _ = kendalltau(predict_arr[:node_num], true_arr[:node_num])

    return kt


def loss_cal(y_out, true_val, num_nodes, device, model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    _, order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes * 20

    ind_1 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)
    ind_2 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)

    rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()

    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)

    return loss_rank