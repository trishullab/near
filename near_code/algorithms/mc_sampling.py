import copy
import random
import time

from .core import ProgramLearningAlgorithm, ProgramNodeFrontier
from program_graph import ProgramGraph, ProgramNode
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train


class MC_SAMPLING(ProgramLearningAlgorithm):

    def __init__(self, num_mc_samples=10):
        self.num_mc_samples = num_mc_samples # number of mc samples before choosing a child

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        current = copy.deepcopy(graph.root_node)
        current_avg_f_score = float('inf')
        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []
        num_children_trained = 0
        start_time = time.time()

        while not graph.is_fully_symbolic(current.program):
            log_and_print("CURRENT program has avg fscore {:.4f}: {}".format(
                current_avg_f_score, print_program(current.program, ignore_constants=(not verbose))))
            children = graph.get_all_children(current, in_enumeration=True)
            children_mapping = { print_program(child.program, ignore_constants=True) : child for child in children }
            children_scores = { key : [] for key in children_mapping.keys() }
            costs = [child.cost for child in children]
            
            for i in range(self.num_mc_samples):
                child = random.choices(children, weights=costs)[0]
                sample = self.mc_sample(graph, child)
                assert graph.is_fully_symbolic(sample.program)
                
                log_and_print("Training sample program: {}".format(print_program(sample.program, ignore_constants=(not verbose))))
                sample_score = execute_and_train(sample.program, validset, trainset, train_config, 
                    graph.output_type, graph.output_size, neural=False, device=device)
                num_children_trained += 1
                log_and_print("{} total children trained".format(num_children_trained))

                sample_f_score = sample.cost + sample_score
                children_scores[print_program(child.program, ignore_constants=True)].append(sample_f_score)

                if sample_f_score < best_total_cost:
                    best_program = copy.deepcopy(sample.program)
                    best_total_cost = sample_f_score
                    best_programs_list.append({
                            "program" : best_program,
                            "struct_cost" : sample.cost,
                            "score" : sample_score,
                            "path_cost" : sample_f_score,
                            "time" : time.time()-start_time
                        })
                    log_and_print("New BEST program found:")
                    print_program_dict(best_programs_list[-1])

            # (Naive) selection operation
            children_scores = { key : sum(val)/len(val) if len(val) > 0 else float('inf') for key,val in children_scores.items() }
            best_child_name = min(children_scores, key=children_scores.get)
            current = children_mapping[best_child_name]
            current_avg_f_score = children_scores[best_child_name]
            for key,val in children_scores.items():
                log_and_print("Avg score {:.4f} for child {}".format(val,key))
            log_and_print("SELECTING {} as best child node\n".format(best_child_name))
            log_and_print("DEBUG: time since start is {:.3f}\n".format(time.time()-start_time))

        return best_programs_list

    def mc_sample(self, graph, program_node):
        assert isinstance(program_node, ProgramNode)
        while not graph.is_fully_symbolic(program_node.program):
            children = graph.get_all_children(program_node, in_enumeration=True)
            costs = [child.cost for child in children]
            program_node = random.choices(children, weights=costs)[0]
        return program_node
