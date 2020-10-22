import copy
import math
import random
import time

from .core import ProgramLearningAlgorithm, ProgramNodeFrontier
from program_graph import ProgramGraph, ProgramNode
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train
from collections import defaultdict


class MCTS(ProgramLearningAlgorithm):

    def __init__(self, num_mc_samples=10, ucb_coeff=math.sqrt(2)):
        self.num_mc_samples = num_mc_samples # number of mc samples before choosing a child
        self.ucb_coeff = ucb_coeff # C coefficient in UCB formula
        self.program_scores = defaultdict(list)

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        current = copy.deepcopy(graph.root_node)
        current_avg_f_score = float('inf')
        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []
        num_programs_trained = 0
        start_time = time.time()

        while not graph.is_fully_symbolic(current.program):
            log_and_print("CURRENT program has avg fscore {:.4f}: {}".format(
                current_avg_f_score, print_program(current.program, ignore_constants=(not verbose))))
            
            for i in range(self.num_mc_samples):
                sample, sample_path = self.mcts_sample(graph, current)
                assert graph.is_fully_symbolic(sample.program)
                
                log_and_print("Training sample program: {}".format(print_program(sample.program, ignore_constants=(not verbose))))
                sample_score = execute_and_train(sample.program, validset, trainset, train_config, 
                    graph.output_type, graph.output_size, neural=False, device=device)
                num_programs_trained += 1
                log_and_print("{} total programs trained".format(num_programs_trained))

                sample_f_score = sample.cost + sample_score

                # Update scores for all programs along path
                for program_name in sample_path:
                    self.program_scores[program_name].append(sample_f_score)

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

            # (Naively) choose best child 
            children = graph.get_all_children(current, in_enumeration=True)
            children_mapping = { print_program(child.program, ignore_constants=True) : child for child in children }
            children_scores = { key : self.program_scores[key] for key in children_mapping.keys() }
            children_scores = { key : sum(val)/len(val) if len(val) > 0 else float('inf') for key,val in children_scores.items() } 
            best_child_name = [key for key,val in children_scores.items() if val==min(children_scores.values())][0] # min score
            current = children_mapping[best_child_name]
            current_avg_f_score = children_scores[best_child_name]
            for key,val in children_scores.items():
                log_and_print("Avg score {:.4f} for child {}".format(val,key))
            log_and_print("SELECTING {} as best child node\n".format(best_child_name))
            log_and_print("DEBUG: time since start is {:.3f}\n".format(time.time()-start_time))

        return best_programs_list

    def mcts_sample(self, graph, program_node):
        assert isinstance(program_node, ProgramNode)
        program_path = []
        while not graph.is_fully_symbolic(program_node.program):
            children = graph.get_all_children(program_node, in_enumeration=True)
            children_mapping = { print_program(child.program, ignore_constants=True) : child for child in children }
            children_scores = { key : self.program_scores[key] for key in children_mapping.keys() }
            child_name = self.ucb_select(children_scores)
            program_node = children_mapping[child_name]
            program_path.append(print_program(program_node.program, ignore_constants=True))
        return program_node, program_path

    def ucb_select(self, children_scores):
        scores = { key : sum(val)/len(val) if len(val) > 0 else float('inf') for key,val in children_scores.items() } 
        count = { key : len(val) for key,val in children_scores.items() } 

        N = sum(count.values()) 
        ucb_vals = {}

        for child in children_scores.keys():
            if count[child] == 0:
                ucb_vals[child] = float('inf')
            else:
                ucb_vals[child] = (1-scores[child]) + self.ucb_coeff*math.sqrt(math.log(N)/count[child])

        # Select child with max UCB weight
        selection = [key for key,val in ucb_vals.items() if val==max(ucb_vals.values())][0]

        return selection
