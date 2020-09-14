import copy
import time

from .core import ProgramLearningAlgorithm, ProgramNodeFrontier
from program_graph import ProgramGraph
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train


class IDDFS_NEAR(ProgramLearningAlgorithm):

    def __init__(self, frontier_capacity=float('inf'), initial_depth=1, performance_multiplier=1.0, depth_bias=1.0, exponent_bias=False):
        self.frontier_capacity = frontier_capacity
        self.initial_depth = initial_depth
        self.performance_multiplier = 1.0 # < 1.0 prunes more aggressively
        self.depth_bias = 1.0 # < 1.0 prunes more aggressively
        self.exponent_bias = exponent_bias # flag to determine it depth_bias should be exponentiated or not

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training root program ...")
        current = copy.deepcopy(graph.root_node)
        initial_score = execute_and_train(current.program, validset, trainset, train_config, 
            graph.output_type, graph.output_size, neural=True, device=device)
        log_and_print("Initial training complete. Score from program is {:.4f} \n".format(1 - initial_score))
        
        # Branch-and-bound search with iterative deepening
        current_depth = self.initial_depth
        current_f_score = float('inf')
        order = 0
        frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
        next_frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
        num_children_trained = 0
        start_time = time.time()

        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []

        log_and_print("Starting iterative deepening with depth {}\n".format(current_depth))

        while current_depth <= graph.max_depth:
            log_and_print("CURRENT program has fscore {:.4f}: {}".format(
                current_f_score, print_program(current.program, ignore_constants=(not verbose))))
            log_and_print("Current depth of program is {}".format(current.depth))
            log_and_print("Creating children for current node/program")
            log_and_print("Total time elapsed is {:.3f}".format(time.time()-start_time))
            children_nodes = graph.get_all_children(current)

            # prune if more than self.max_num_children
            if len(children_nodes) > graph.max_num_children:
                log_and_print("Sampling {}/{} children".format(graph.max_num_children, len(children_nodes)))
                children_nodes = random.sample(children_nodes, k=graph.max_num_children) # sample without replacement
            log_and_print("{} total children to train for current node".format(len(children_nodes)))
            
            child_tuples = []
            for child_node in children_nodes:
                child_start_time = time.time()
                log_and_print("Training child program: {}".format(print_program(child_node.program, ignore_constants=(not verbose))))
                is_neural = not graph.is_fully_symbolic(child_node.program)
                child_node.score = execute_and_train(child_node.program, validset, trainset, train_config, 
                    graph.output_type, graph.output_size, neural=is_neural, device=device)
                log_and_print("Time to train child {:.3f}".format(time.time()-child_start_time))
                num_children_trained += 1
                log_and_print("{} total children trained".format(num_children_trained))
                child_node.parent = current
                child_node.children = []
                order -= 1
                child_node.order = order # insert order of exploration as tiebreaker for equivalent f-scores

                # computing path costs (f_scores)
                child_f_score = child_node.cost + child_node.score # cost + heuristic
                log_and_print("DEBUG: f-score {}".format(child_f_score))
                current.children.append(child_node)
                child_tuples.append((child_f_score, order, child_node))

                if not is_neural and child_f_score < best_total_cost:
                    best_program = copy.deepcopy(child_node.program)
                    best_total_cost = child_f_score
                    best_programs_list.append({
                            "program" : best_program,
                            "struct_cost" : child_node.cost, 
                            "score" : child_node.score,
                            "path_cost" : child_f_score,
                            "time" : time.time()-start_time
                        })
                    log_and_print("New BEST program found:")
                    print_program_dict(best_programs_list[-1])

            # find next current among children, from best to worst
            nextfound = False
            child_tuples.sort(key=lambda x: x[0])
            for child_tuple in child_tuples:
                child = child_tuple[2]
                if graph.is_fully_symbolic(child.program):
                    continue # don't want to expand symbolic programs (no children)
                elif child.depth >= current_depth:
                    next_frontier.add(child_tuple)
                else:
                    if not nextfound:
                        nextfound = True # first child program that's not symbolic and within current_depth
                        current_f_score, current_order, current = child_tuple
                        log_and_print("Found program among children: {} with f_score {}".format(
                            print_program(current.program, ignore_constants=(not verbose)), current_f_score))
                    else:
                        frontier.add(child_tuple) # put the rest onto frontier

            # find next node in frontier
            if not nextfound:
                frontier.sort(tup_idx=1) # DFS order
                log_and_print("Frontier length is: {}".format(len(frontier)))
                original_depth = current.depth
                while len(frontier) > 0 and not nextfound:
                    current_f_score, current_order, current = frontier.pop(0, sort_fscores=False) # DFS order
                    if current_f_score < self.bound_modify(best_total_cost, original_depth, current.depth):
                        nextfound = True
                        log_and_print("Found program in frontier: {} with f_score {}".format(
                            print_program(current.program, ignore_constants=(not verbose)), current_f_score))
                    else:
                        log_and_print("PRUNE from frontier: {} with f_score {}".format(
                            print_program(current.program, ignore_constants=(not verbose)), current_f_score))
                log_and_print("Frontier length is now {}".format(len(frontier)))

            # frontier is empty, go to next stage of iterative deepening
            if not nextfound:
                assert len(frontier) == 0
                log_and_print("Empty frontier, moving to next depth level")
                log_and_print("DEBUG: time since start is {:.3f}\n".format(time.time() - start_time))

                current_depth += 1

                if current_depth > graph.max_depth:
                    log_and_print("Max depth {} reached. Exiting.\n".format(graph.max_depth))
                    break
                elif len(next_frontier) == 0:
                    log_and_print("Next frontier is empty. Exiting.\n")
                    break
                else:
                    log_and_print("Starting iterative deepening with depth {}\n".format(current_depth))
                    frontier = copy.deepcopy(next_frontier)
                    next_frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
                    current_f_score, current_order, current = frontier.pop(0)

        if best_program is None:
            log_and_print("ERROR: no program found")

        return best_programs_list

    def bound_modify(self, upperbound, current_depth, node_depth):
        if not self.exponent_bias:
            depth_multiplier = self.performance_multiplier * (self.depth_bias**(current_depth-node_depth))
        else:
            depth_multiplier = self.performance_multiplier ** (self.depth_bias**(current_depth-node_depth))
        return upperbound * depth_multiplier
