import copy

from .core import ProgramLearningAlgorithm
from program_graph import ProgramGraph
from utils.logging import log_and_print
from utils.training import execute_and_train


class RNN_BASELINE(ProgramLearningAlgorithm):

    def __init__(self):
        log_and_print("Root node is Start(ListToListModule) or Start(ListToAtomModule), both implemented with an RNN.")
        log_and_print("Be sure to set neural_epochs and max_num_units accordingly.\n")

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training RNN baseline with {} LSTM units ...".format(graph.max_num_units))
        current = copy.deepcopy(graph.root_node)
        score = execute_and_train(current.program, validset, trainset, train_config, 
            graph.output_type, graph.output_size, neural=True, device=device)
        log_and_print("Score of RNN is {:.4f}".format(1-score))
        
        return current.program
