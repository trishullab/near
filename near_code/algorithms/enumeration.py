import copy
import time

from .core import ProgramLearningAlgorithm
from program_graph import ProgramGraph
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train


class ENUMERATION(ProgramLearningAlgorithm):

    def __init__(self, max_num_programs=100):
        self.max_num_programs = max_num_programs

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        symbolic_programs = []
        enum_depth = 1
        while len(symbolic_programs) < self.max_num_programs:
            print("DEBUG: starting enumerative synthesis with depth {}".format(enum_depth))
            symbolic_programs = self.enumerate2depth(graph, enum_depth)
            print("DEBUG: {} programs found".format(len(symbolic_programs)))
            enum_depth += 1
            if enum_depth > graph.max_depth:
                break
        log_and_print("Symbolic Synthesis: generated {}/{} symbolic programs from candidate program.".format(
            len(symbolic_programs), self.max_num_programs))
        
        total_eval = min(self.max_num_programs, len(symbolic_programs))
        symbolic_programs.sort(key=lambda x: x["struct_cost"])
        symbolic_programs = symbolic_programs[:total_eval]

        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []
        start_time = time.time()
        num_programs_trained = 1

        for prog_dict in symbolic_programs:
            child_start_time = time.time()
            candidate = prog_dict["program"]
            log_and_print("Training candidate program ({}/{}) {}".format(
                num_programs_trained, total_eval, print_program(candidate, ignore_constants=(not verbose))))
            num_programs_trained += 1
            score = execute_and_train(candidate, validset, trainset, train_config, 
                graph.output_type, graph.output_size, neural=False, device=device)

            total_cost = score + prog_dict["struct_cost"]
            log_and_print("Structural cost is {} with structural penalty {}".format(prog_dict["struct_cost"], graph.penalty))
            log_and_print("Time to train child {:.3f}".format(time.time()-child_start_time))
            log_and_print("Total time elapsed is: {:.3f}".format(time.time()-start_time))

            if total_cost < best_total_cost:
                best_program = copy.deepcopy(prog_dict["program"])
                best_total_cost = total_cost
                prog_dict["score"] = score
                prog_dict["path_cost"] = total_cost
                prog_dict["time"] = time.time()-start_time
                best_programs_list.append(prog_dict)
                log_and_print("New BEST program found:")
                print_program_dict(best_programs_list[-1])

        return best_programs_list

    def enumerate2depth(self, graph, enumeration_depth):
        max_depth_copy = graph.max_depth
        graph.max_depth = enumeration_depth
        all_programs = []
        enumerated = {}
        root = copy.deepcopy(graph.root_node)
        
        def enumerate_helper(program_node):
            program_name = print_program(program_node.program, ignore_constants=True)
            assert not enumerated.get(program_name)
            enumerated[program_name] = True
            if graph.is_fully_symbolic(program_node.program):
                all_programs.append({
                        "program" : copy.deepcopy(program_node.program),
                        "struct_cost" : program_node.cost,
                        "depth" : program_node.depth
                    })
            elif program_node.depth < enumeration_depth:
                all_children = graph.get_all_children(program_node, in_enumeration=True)
                for childnode in all_children:
                    if not enumerated.get(print_program(childnode.program, ignore_constants=True)):
                        enumerate_helper(childnode)
        
        enumerate_helper(root)
        graph.max_depth = max_depth_copy

        return all_programs
