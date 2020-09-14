import torch
from .neural_functions import init_neural_function
from .library_functions import LibraryFunction


# TODO allow user to choose device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


class RunningAverageFunction(LibraryFunction):
    """Computes running average over a window, then applies an Atom2AtomModule on the average."""

    def __init__(self, input_size, output_size, num_units, a2a_function=None, name="RunningAvg"):
        if a2a_function is None:
            a2a_function = init_neural_function("atom", "atom", input_size, output_size, num_units)
        submodules = { "subfunction" : a2a_function }
        super().__init__(submodules, "list", "atom", input_size, output_size, num_units, name=name,)

    def window_start(self, t):
        return 0

    def window_end(self, t):
        return t

    def execute_on_batch(self, batch, batch_lens, is_sequential=False):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        batch = batch.transpose(0,1) # (seq_len, batch_size, feature_dim)

        out = []
        for t in range(seq_len):
            window_start = max(0, self.window_start(t))
            window_end = min(seq_len, self.window_end(t))
            window = batch[window_start:window_end+1]
            running_average = torch.mean(window, dim=0)
            out_val = self.submodules["subfunction"].execute_on_batch(running_average)
            out.append(out_val.unsqueeze(1))
        out = torch.cat(out, dim=1)
        
        if not is_sequential:
            idx = torch.tensor(batch_lens).to(device) - 1
            idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out.size(-1))
            out = out.gather(1, idx).squeeze(1)

        return out

class RunningAverageLast5Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None):
        super().__init__(input_size, output_size, num_units, a2a_function, name="Last5Avg")

    def window_start(self, t):
        return t-4

    def window_end(self, t):
        return t

class RunningAverageLast10Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None):
        super().__init__(input_size, output_size, num_units, a2a_function, name="Last10Avg")

    def window_start(self, t):
        return t-9

    def window_end(self, t):
        return t

class RunningAverageWindow11Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None):
        super().__init__(input_size, output_size, num_units, a2a_function, name="Window7Avg")

    def window_start(self, t):
        return t-5

    def window_end(self, t):
        return t+5    
    
class RunningAverageWindow7Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None):
        super().__init__(input_size, output_size, num_units, a2a_function, name="Window7Avg")

    def window_start(self, t):
        return t-3

    def window_end(self, t):
        return t+3
    
class RunningAverageWindow5Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None):
        super().__init__(input_size, output_size, num_units, a2a_function, name="Window5Avg")

    def window_start(self, t):
        return t-2

    def window_end(self, t):
        return t+2    
        
