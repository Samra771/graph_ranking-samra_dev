from graph_mgmt_library import Graph
from betweennes_model import GNN_Bet
import graph_utils
import torch
import networkx as nx 
from betweennes_training_library import *
GRAPH_NODES_NUMBER = 200
GRAPH_SPARSENESS = 0.01
GRAPHS_NUMBER = 100
MODEL_SIZE = GRAPH_NODES_NUMBER

# Define the device to be used for tensor operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':


    print("########################################################################")
    print("#                      GRAPH RANKING PREDICTION                        #")
    print("########################################################################")

    print("Select graph type: ")
    print("1 - Barabàsi-Albert")
    print("2 - Erdòs-Renyi")
    graph_type = input("Your choice? [1] : ")
    if not graph_type:
        graph_type = 1
    graph_type = int(graph_type)

    graph_nodes_number = int(input("Number of nodes in graph? [200]: "))
    if not graph_nodes_number:
        graph_nodes_number = 200
    graph_nodes_number = int(graph_nodes_number)

    graph_sparseness = float(input("Graph sparseness? [0.01]: "))
    if not graph_sparseness:
        graph_sparseness = 0.01

    hidden_layers = int(input("Number of hidden layers [20]: "))
    if not hidden_layers:
        hidden_layers = 20
    hidden_layers = int(hidden_layers)

    learning_rate = float(input("Learning Rate [0.0005]: "))
    if not learning_rate:
        learning_rate = 0.0005
    learning_rate = float(learning_rate)

    dropout = float(input("dropout [0.2]: "))
    if not dropout:
        dropout = 0.2
    dropout = float(dropout)

    weight_decay = float(input("Weight decay  [0.01]: "))
    if not weight_decay:
        weight_decay = 0.01
    weight_decay = float(weight_decay)

    MODEL_SIZE = graph_nodes_number

    print("")
    print("... Creating graph ...")
    G = Graph(graph_nodes_number, graph_sparseness)
    graph = G.create_graph(graph_type)
    adj_matrix = G.get_full_adjacency_matrix(graph)
    adj_matrix_T = adj_matrix.transpose()
    betweenness = G.get_betweenness_centrality(graph)
    num_nodes = len(G.get_graph_nodes(graph)) 
    
    print(f"Select model:")

    models_list = graph_utils.get_files_with_prefix('betweennes')

    for index, filename in models_list.items():
        print(f"{index} - {filename}")

    model_index = int(input("Your choice? [1] : "))
    if not model_index:
        model_index = 1

    print("... Loading Model ...")
    model_name = models_list[model_index]
    network = GNN_Bet(ninput=MODEL_SIZE, nhid=hidden_layers, dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay)
    model, device = network.model_to_device(network)
    model.load_state_dict(torch.load(model_name, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, device = network.model_to_device(network)
    model.load_state_dict(torch.load(model_name, map_location=device))
    num_epochs = 15
    model.eval()


    adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix, device)
    adj_matrix_T = sparse_mx_to_torch_sparse_tensor(adj_matrix_T, device)

    output = model(adj_matrix, adj_matrix_T)

# TRAIN FUNCTION
# --------------------
def train(adj_list, adj_t_list, node_counts, targets):
    model.train()
    loss_train = 0
    for i in range(len(adj_list)):
        adj_matrix = adj_list[i].to(device)
        adj_matrix_T = adj_t_list[i].to(device)

        optimizer.zero_grad()
        pred = model(adj_matrix, adj_matrix_T)
        # Convert targets to tensor
        targets = np.array(targets)
        target = torch.from_numpy(targets[:, i]).float().to(device)
        loss = loss_cal(pred, target, node_counts[i], device, MODEL_SIZE)

        loss.backward()
        optimizer.step()
        loss_train += float(loss)
    return loss_train / len(adj_list)
# --------------------
# TEST FUNCTION
# --------------------
def test(adj_list, adj_t_list, node_counts, targets):
    model.eval()
    kt_scores = []

    with torch.no_grad():
        for i in range(len(adj_list)):
            adj_matrix = adj_list[i].to(device)
            adj_matrix_T = adj_t_list[i].to(device)
            target = torch.from_numpy(targets[:, i]).float().to(device)

            pred = model(adj_matrix, adj_matrix_T)
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            pred = pred.reshape((MODEL_SIZE))
            target = target.reshape((MODEL_SIZE))
            kt = ranking_correlation(pred, target, node_counts[i], MODEL_SIZE)
            kt_scores.append(kt)

    avg_kt = np.mean(kt_scores)
    std_kt = np.std(kt_scores)
    print(f"   Average KT score on test graphs: {avg_kt:.4f}, Std: {std_kt:.4f}")



    


    


