from graph_mgmt_library import Graph
from betweennes_model import GNN_Bet
from torch.utils.tensorboard import SummaryWriter
import betweennes_training_library as training
import graph_utils
import time

TRAIN_SIZE = 0.8

log_dir = f"./logs/run_{time.strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

if __name__=='__main__':

    print("########################################################################")
    print("#                      GRAPH RANKING TRAINING                          #")
    print("########################################################################")
    print("")
    print("                         Graph Parameters")
    print("Graph type: ")
    print("1 - Barabàsi-Albert")
    print("2 - Erdòs-Renyi")
    print("9 - Random")
    graph_type = input("Your choice? [2] : ")
    if not graph_type:
        graph_type = 2
    graph_type = int(graph_type)
    random_graph = False
    if graph_type ==9:
        random_graph = True

    graph_nodes_number = input("Number of nodes in graph? [200]: ")
    if not graph_nodes_number:
        graph_nodes_number = 200
    graph_nodes_number = int(graph_nodes_number)

    graph_sparseness = input("Graph sparseness? [0.01]: ")
    if not graph_sparseness:
        graph_sparseness = 0.01
    graph_sparseness = float(graph_sparseness)

    graphs_number = input("Number of graphs to be generated? [200]: ")
    if not graphs_number:
        graphs_number = 200
    graphs_number = int(graphs_number)

    G = Graph(graph_nodes_number, graph_sparseness)

    print("")
    print("                      Network Parameters")
    hidden_layers = input("Number of hidden layers [20]: ")
    if not hidden_layers:
        hidden_layers = 20
    hidden_layers = int(hidden_layers)

    epoch_number = input("Epoch number [100]: ")
    if not epoch_number:
        epoch_number = 100
    epoch_number = int(epoch_number)

    learning_rate = input("Learning Rate [0.0005]: ")
    if not learning_rate:
        learning_rate = 0.0005
    learning_rate = float(learning_rate)

    dropout = input("dropout [0.2]: ")
    if not dropout:
        dropout = 0.2
    dropout = float(dropout)

    weight_decay = input("Weight decay  [0.01]: ")
    if not weight_decay:
        weight_decay = 0.01
    weight_decay = float(weight_decay)

    batch_size = input("Batch size [16]: ")
    if not batch_size:
        batch_size = 16
    batch_size = int(batch_size)

    save_model = input("Save model? (y/n)[n]: ")
    if not save_model:
        save_model = 'n'

    MODEL_SIZE = graph_nodes_number
    print("")

    graphs = []
    adj_matrices = []
    adj_matrices_T = []
    graphs_betweennesses = []
    graphs_nodes = []

    # Creates graph dataset
    for i in range(0, graphs_number):

        if random_graph:
            graph_type = graph_utils.get_random_graph(1, 2)

        graph = G.create_graph(graph_type)
        graphs.append(graph)

        adj_matrix = G.get_full_adjacency_matrix(graph)
        adj_matrices.append(adj_matrix)

        adj_matrix_T = adj_matrix.transpose()
        adj_matrices_T.append(adj_matrix_T)

        betweenness = G.get_betweenness_centrality(graph)
        graphs_betweennesses.append(betweenness)

        num_nodes = len(G.get_graph_nodes(graph))
        graphs_nodes.append(num_nodes)

        if i % 10 == 0:
            print(f"Created {i} graphs")

    print(f"Train/Test splitting")
    # Splits test and train datasets
    indices = graph_utils.shuffle_graphs(graphs_number)
    train_indices = graph_utils.get_random_sample(indices, TRAIN_SIZE)
    adj_matrices_train, adj_matrices_test = graph_utils.split_train_test_matrices(adj_matrices, train_indices)
    adj_matrices_T_train, adj_matrices_T_test = graph_utils.split_train_test_matrices(adj_matrices_T, train_indices)
    graphs_betweennesses_train, graphs_betweennesses_test = graph_utils.split_train_test_list(graphs_betweennesses, train_indices)
    graphs_nodes_train, graphs_nodes_test = graph_utils.split_train_test_list(graphs_nodes, train_indices)

    print(f"Model creation")
    network = GNN_Bet(ninput=MODEL_SIZE, nhid=hidden_layers, dropout=dropout, learning_rate=learning_rate,
                      weight_decay=weight_decay)
    model, device = network.model_to_device(network)
    optimizer = network.get_optimizer(model)

    print(f"Training")
    for e in range(epoch_number):
        print(f"Training epoch {e} of {epoch_number}")
        train_loss = training.train(adj_matrices_train, adj_matrices_T_train, graphs_nodes_train, graphs_betweennesses_train,
                       model, device, optimizer, MODEL_SIZE, writer, e, batch_size=batch_size)

        test_loss = training.test(adj_matrices_test, adj_matrices_T_test, graphs_nodes_test, graphs_betweennesses_test,
                       model, device, optimizer, MODEL_SIZE, writer, e)

        relative_loss_diff = (train_loss - test_loss) / train_loss
        writer.add_scalar("Loss/relative_diff", relative_loss_diff, e)

    print(f"Training completed")

    if save_model == 'y':
        network.save_model(model)

    writer.close()


