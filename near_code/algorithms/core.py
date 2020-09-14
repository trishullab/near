from program_graph import ProgramNode
from utils.logging import log_and_print, print_program


class ProgramLearningAlgorithm(object):
    
    def __init__(self, **kwargs):
        pass

    def run(self, **kwargs):
        raise NotImplementedError


class ProgramNodeFrontier(object):
    
    def __init__(self, capacity=float('inf')):
        self.capacity = capacity
        self.prioqueue = []

    def __len__(self):
        return len(self.prioqueue)

    def add(self, item):
        assert len(item) == 3
        assert isinstance(item[2], ProgramNode)
        self.prioqueue.append(item)
        if len(self.prioqueue) > self.capacity:
            # self.sort(tup_idx=0)
            popped_f_score, _, popped = self.pop(-1)
            log_and_print("POP {} with fscore {:.4f}".format(print_program(popped.program, ignore_constants=True), popped_f_score))

    def peek(self, idx=0):
        if len(self.prioqueue) == 0:
            return None
        return self.prioqueue[idx]

    def pop(self, idx, sort_fscores=True):
        """Pops the first item off the queue."""
        if len(self.prioqueue) == 0:
            return None
        if sort_fscores:
            self.sort(tup_idx=0)
        return self.prioqueue.pop(idx)

    def sort(self, tup_idx=0):
        self.prioqueue.sort(key=lambda x: x[tup_idx])
