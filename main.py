import logging

import torch

from exp.exp_graph_partition import ExpGraphPartition
from exp.exp_node_edge_unlearning import ExpNodeEdgeUnlearning
from exp.exp_unlearning import ExpUnlearning
from exp.exp_attack_unlearning import ExpAttackUnlearning
from parameter_parser import parameter_parser


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main(args, exp):
    # config the logger
    logger_name = "_".join((exp, args['dataset_name'], args['partition_method'], str(args['num_shards']), str(args['test_ratio'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])

    # subroutine entry for different methods
    if exp == 'partition':
        ExpGraphPartition(args)
    elif exp == 'unlearning':
        ExpUnlearning(args)
    elif exp == 'node_edge_unlearning':
        ExpNodeEdgeUnlearning(args)
    elif exp == 'attack_unlearning':
        ExpAttackUnlearning(args)
    else:
        raise Exception('unsupported attack')


if __name__ == "__main__":
    args = parameter_parser()

    main(args, args['exp'])
