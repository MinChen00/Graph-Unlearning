import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        choices=["cora", "citeseer", "pubmed", "Coauthor_CS", "Coauthor_Phys"])

    parser.add_argument('--exp', type=str, default='attack_unlearning',
                        choices=["partition", "unlearning", "node_edge_unlearning", "attack_unlearning"])
    parser.add_argument('--cuda', type=int, default=3, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)

    parser.add_argument('--is_upload', type=str2bool, default=True)
    parser.add_argument('--database_name', type=str, default="unlearning_dependant",
                        choices=['unlearning_dependant', 'unlearning_adaptive',
                                 'unlearning_graph_structure', 'gnn_unlearning_shards',
                                 'unlearning_delta_plot', 'gnn_unlearning_utility',
                                 'unlearning_ratio', 'unlearning_partition_baseline',
                                 'unlearning_ratio', 'attack_unlearning'])

    ########################## graph partition parameters ######################
    parser.add_argument('--is_split', type=str2bool, default=True)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    parser.add_argument('--is_partition', type=str2bool, default=True)
    parser.add_argument('--is_prune', type=str2bool, default=False)
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--is_constrained', type=str2bool, default=True)
    parser.add_argument('--is_gen_embedding', type=str2bool, default=True)

    parser.add_argument('--partition_method', type=str, default='sage_km',
                        choices=["sage_km", "random", "lpa", "metis", "lpa_base", "sage_km_base"])
    parser.add_argument('--terminate_delta', type=int, default=0)
    parser.add_argument('--shard_size_delta', type=float, default=0.005)

    ########################## unlearning parameters ###########################
    parser.add_argument('--repartition', type=str2bool, default=False)

    ########################## training parameters ###########################
    parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    parser.add_argument('--is_use_node_feature', type=str2bool, default=False)
    parser.add_argument('--is_use_batch', type=str2bool, default=True, help="Use batch train GNN models.")
    parser.add_argument('--target_model', type=str, default='GAT', choices=["SAGE", "GAT", 'MLP', "GCN", "GIN"])
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--aggregator', type=str, default='mean', choices=['mean', 'majority', 'optimal'])

    parser.add_argument('--opt_lr', type=float, default=0.001)
    parser.add_argument('--opt_decay', type=float, default=0.0001)
    parser.add_argument('--opt_num_epochs', type=int, default=50)
    parser.add_argument('--unlearning_request', type=str, default='random', choices=['random', 'adaptive', 'dependant', 'top1', 'last5'])

    ########################## analysis parameters ###################################
    parser.add_argument('--num_unlearned_nodes', type=int, default=1)
    parser.add_argument('--ratio_unlearned_nodes', type=float, default=0.005)
    parser.add_argument('--num_unlearned_edges', type=int, default=1)
    parser.add_argument('--ratio_deleted_edges', type=float, default=0.9)
    parser.add_argument('--num_opt_samples', type=int, default=1000)

    args = vars(parser.parse_args())

    return args
