import copy
import random
import time

import dsl
from .core import ProgramLearningAlgorithm
from program_graph import ProgramGraph, ProgramNode
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train


class GENETIC(ProgramLearningAlgorithm):

    def __init__(self, population_size=15, selection_size=8, num_gens=20, total_eval=100, mutation_prob=.1, max_enum_depth=6):
        self.population_size = population_size
        self.selection_size = selection_size
        self.num_gens = num_gens
        self.total_eval = total_eval
        self.mutation_prob = mutation_prob
        self.max_enum_depth = max_enum_depth

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)
        self.input_size = graph.input_size
        self.output_size = graph.output_size

        best_program = None
        best_score = float('inf')
        best_total = float('inf')
        best_programs_list = []
        start_time = time.time()

        log_and_print("Generating all programs up to specified depth.")
        all_generated_programs = self.enumerative_synthesis(graph, self.max_enum_depth)
        random.shuffle(all_generated_programs)
        current_population = all_generated_programs[:self.population_size]
        total_trained_programs = 0

        def crossover_operation(listofprogramdicts):
            new_prog_dict = {}
            parents = random.choices(listofprogramdicts, k=2)
            prog_dict1, prog_dict2 = parents[0], parents[1]
            xover1 = random.randint(1, prog_dict1['depth'])
            xover2 = random.randint(2, prog_dict2['depth'])
            print('DEBUG: trying new crossover')
            new_program, crossed_over = self.crossover_helper(prog_dict1['program'], prog_dict2['program'], xover1, xover2)

            if crossed_over:
                log_and_print("Crossing over the following programs:")
                log_and_print(print_program(prog_dict1['program'], ignore_constants=True))
                log_and_print(print_program(prog_dict2['program'], ignore_constants=True))
                new_prog_dict["program"] = new_program
                new_prog_dict["struct_cost"], new_prog_dict["depth"] = graph.compute_program_cost(new_program)
                log_and_print("Result has structural cost {:.4f}:".format(new_prog_dict["struct_cost"]))
                log_and_print(print_program(new_prog_dict['program'], ignore_constants=True))
                return new_prog_dict
            else:
                return None

        def mutation_operation(mod_prog_dict):
            new_prog_dict = {}
            mutation_point = random.randrange(1, mod_prog_dict['depth'])
            new_prog_dict['program'] = self.mutation_helper(graph, mod_prog_dict['program'], mod_prog_dict['depth'],
                                                            mutation_point, max_enumeration_depth=self.max_enum_depth)
            new_prog_dict["struct_cost"], new_prog_dict["depth"] = graph.compute_program_cost(new_prog_dict['program'])
            return new_prog_dict

        for gen_idx in range(self.num_gens):
            #evaluation operation: train each program and evaluate it.
            log_and_print("Training generation {}'s population of programs.".format(gen_idx + 1))
            for programdict in current_population:
                total_trained_programs += 1
                child_start_time = time.time()
                log_and_print("Training candidate program ({}/{}) {}".format(
                    total_trained_programs, self.total_eval,
                    print_program(programdict['program'], ignore_constants=(not verbose))))
                score =  execute_and_train(programdict['program'], validset, trainset, train_config, 
                    graph.output_type, graph.output_size, neural=False, device=device)
                log_and_print(
                    "Structural cost is {} with structural penalty {}".format(programdict["struct_cost"], graph.penalty))
                log_and_print("Time to train child {:.3f}".format(time.time() - child_start_time))
                log_and_print("Total time elapsed is: {}".format(time.time() - start_time))
                programdict["score"] = score
                programdict["path_cost"] = score + programdict["struct_cost"]
                programdict["time"] = time.time() - start_time
                if programdict["path_cost"] < best_total:
                    best_program = copy.deepcopy(programdict["program"])
                    best_total = score + programdict["struct_cost"]
                    best_score = score
                    best_cost = programdict["struct_cost"]
                    best_programs_list.append(copy.deepcopy(programdict))
                    log_and_print("New BEST program found:")
                    print_program_dict(best_programs_list[-1])
            if total_trained_programs > self.total_eval:
                break
            #select the best programs based on cost + score.
            current_population.sort(key=lambda x: x["path_cost"])
            selected_population = current_population[:self.selection_size]
            #perform crossover on the selected population
            crossed_population = []
            log_and_print("Beginning crossover operation.")
            while len(crossed_population) < self.population_size:
                new_prog_dict = crossover_operation(selected_population)
                if new_prog_dict is not None:
                    crossed_population.append(new_prog_dict)
            #perform mutations on the crossed population
            current_population = []
            for crossed_prog_dict in crossed_population:
                if random.random() < self.mutation_prob:
                    log_and_print("Mutating program.")
                    current_population.append(mutation_operation(crossed_prog_dict))
                else:
                    current_population.append(crossed_prog_dict)

        return best_programs_list

    def enumerative_synthesis(self, graph, enumeration_depth, typesig=None, input_size=None, output_size=None):
        #construct the num_selected lists
        max_depth_copy = graph.max_depth
        graph.max_depth = enumeration_depth
        all_programs = []
        enumerated = {}
        input_size = self.input_size if input_size is None else input_size
        output_size = self.output_size if output_size is None else output_size
        if typesig is None:
            root = copy.deepcopy(graph.root_node)
        else:
            new_start = dsl.StartFunction(input_type=typesig[0], output_type=typesig[1], input_size=input_size, output_size=output_size, num_units=graph.max_num_units)
            root = ProgramNode(new_start, 0, None, 0, 0, 0)
        def enumerate_helper(currentnode):
            printedprog = print_program(currentnode.program, ignore_constants=True)
            assert not enumerated.get(printedprog)
            enumerated[printedprog] = True
            if graph.is_fully_symbolic(currentnode.program):
                all_programs.append({
                        "program" : copy.deepcopy(currentnode.program),
                        "struct_cost" : currentnode.cost,
                        "depth" : currentnode.depth
                    })
            elif currentnode.depth < enumeration_depth:
                all_children = graph.get_all_children(currentnode, in_enumeration=True)
                for childnode in all_children:
                    if not enumerated.get(print_program(childnode.program, ignore_constants=True)):
                        enumerate_helper(childnode)
        enumerate_helper(root)
        graph.max_depth = max_depth_copy
        return all_programs

    def find_type_sig(self, program, type_sig, xover_point_2):
        pointer = 1
        queue = [program]
        while len(queue) != 0:
            current_function = queue.pop()
            for submodule, functionclass in current_function.submodules.items():
                if pointer == xover_point_2:
                    if (functionclass.input_type, functionclass.output_type) == type_sig:
                        return functionclass
                    else:
                        xover_point_2 += 1
                pointer += 1
                queue.append(functionclass)
        return None

    def reset_program_sizes(self, program):
        queue = [program]
        program.input_size = self.input_size
        program.output_size = self.output_size
        while len(queue) != 0:
            current_function = queue.pop()
            if isinstance(current_function, dsl.FoldFunction):
                new_input_size = current_function.input_size + current_function.output_size
            else:
                new_input_size = current_function.input_size
            new_output_size = current_function.output_size
            if current_function.has_params:
                current_function.init_params()
            for submodule, functionclass in current_function.submodules.items():
                functionclass.input_size = new_input_size
                if isinstance(current_function, dsl.SimpleITE) and submodule == "evalfunction":
                    functionclass.output_size = 1
                else:
                    functionclass.output_size = new_output_size
                queue.append(functionclass)

    def crossover_helper(self, program1, program2, xover_point_1, xover_point_2):
        pointer = 1
        queue = [program1]
        while len(queue) != 0:
            current_function = queue.pop()
            for submodule, functionclass in current_function.submodules.items():
                if pointer == xover_point_1:
                    replacement = self.find_type_sig(program2,
                                       (functionclass.input_type, functionclass.output_type),
                                       xover_point_2)
                    if replacement is not None:
                        current_function.submodules[submodule] = copy.deepcopy(replacement)
                        self.reset_program_sizes(program1)
                        return program1, True
                    else:
                        xover_point_1 += 1
                pointer += 1
                queue.append(functionclass)
        return program1, False

    def mutation_helper(self, graph, program, total_depth, mutation_point, max_enumeration_depth=6):
        pointer = 1
        queue = [program]
        while len(queue) != 0:
            current_function = queue.pop()
            for submodule, functionclass in current_function.submodules.items():
                if pointer == mutation_point:
                    log_and_print("Generating replacements for mutation program at depth {} and typesig {}.".format(
                                        total_depth - mutation_point, (functionclass.input_type, functionclass.output_type)))
                    enumeration_depth = min(max_enumeration_depth, total_depth - mutation_point)
                    all_replacements = self.enumerative_synthesis(graph, enumeration_depth,
                                                                  (functionclass.input_type, functionclass.output_type),
                                                                  input_size=functionclass.input_size, output_size=functionclass.output_size)
                    if len(all_replacements) == 0:
                        return program
                    else:
                        mutated_replacement = random.choice(all_replacements)
                        current_function.submodules[submodule] = copy.deepcopy(mutated_replacement['program'].submodules['program'])
                        self.reset_program_sizes(program)
                        log_and_print("Mutation completed.")
                        return program
                pointer += 1
                queue.append(functionclass)
        return program
