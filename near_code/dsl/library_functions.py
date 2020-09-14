import torch
import torch.nn as nn

from .neural_functions import init_neural_function


# TODO allow user to choose device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


class LibraryFunction:

    def __init__(self, submodules, input_type, output_type, input_size, output_size, num_units, name="", has_params=False):
        self.submodules = submodules
        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_units = num_units
        self.name = name
        self.has_params = has_params

        if self.has_params:
            assert "init_params" in dir(self)
            self.init_params()

    def get_submodules(self):
        return self.submodules

    def set_submodules(self, new_submodules):
        self.submodules = new_submodules

    def get_typesignature(self):
        return self.input_type, self.output_type

class StartFunction(LibraryFunction):

    def __init__(self, input_type, output_type, input_size, output_size, num_units):
        self.program = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        submodules = { 'program' : self.program }
        super().__init__(submodules, input_type, output_type, input_size, output_size, num_units, name="Start")

    def execute_on_batch(self, batch, batch_lens=None, batch_output=None, is_sequential=False):
        return self.submodules["program"].execute_on_batch(batch, batch_lens)
            
class FoldFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, fold_function=None):
        #TODO: will this accumulator require a grad?
        self.accumulator = torch.zeros(output_size)
        if fold_function is None:
            fold_function = init_neural_function("atom", "atom", input_size+output_size, output_size, num_units)
        submodules = { "foldfunction" : fold_function }
        super().__init__(submodules, "list", "atom", input_size, output_size, num_units, name="Fold")

    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        batch = batch.transpose(0,1) # (seq_len, batch_size, feature_dim)

        fold_out = []
        folded_val = self.accumulator.clone().detach().requires_grad_(True)
        folded_val = folded_val.unsqueeze(0).repeat(batch_size,1).to(device)
        for t in range(seq_len):
            features = batch[t]
            out_val = self.submodules["foldfunction"].execute_on_batch(torch.cat([features, folded_val], dim=1))
            fold_out.append(out_val.unsqueeze(1))
            folded_val = out_val
        fold_out = torch.cat(fold_out, dim=1)
        
        if not is_sequential:
            idx = torch.tensor(batch_lens).to(device) - 1
            idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, fold_out.size(-1))
            fold_out = fold_out.gather(1, idx).squeeze(1)

        return fold_out

class MapFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, map_function=None):
        if map_function is None:
            map_function = init_neural_function("atom", "atom", input_size, output_size, num_units)
        submodules = { "mapfunction" : map_function }
        super().__init__(submodules, "list", "list", input_size, output_size, num_units, name="Map")

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        map_input = batch.view(-1, feature_dim)
        map_output = self.submodules["mapfunction"].execute_on_batch(map_input)
        return map_output.view(batch_size, seq_len, -1)

class MapPrefixesFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, map_function=None):
        if map_function is None:
            map_function = init_neural_function("list", "atom", input_size, output_size, num_units)
        submodules = { "mapfunction" : map_function }
        super().__init__(submodules, "list", "list", input_size, output_size, num_units, name="MapPrefixes")

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        map_output = self.submodules["mapfunction"].execute_on_batch(batch, batch_lens, is_sequential=True)
        assert len(map_output.size()) == 3
        return map_output

class ITE(LibraryFunction):
    """(Smoothed) If-The-Else."""

    def __init__(self, input_type, output_type, input_size, output_size, num_units, eval_function=None, function1=None, function2=None, beta=1.0, name="ITE", simple=False):
        if eval_function is None:
            if simple:
                eval_function = init_neural_function(input_type, "atom", input_size, 1, num_units)
            else:
                eval_function = init_neural_function(input_type, "atom", input_size, output_size, num_units)
        if function1 is None:
            function1 = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        if function2 is None:
            function2 = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        submodules = { "evalfunction" : eval_function, "function1" : function1, "function2" : function2 }
        self.bsmooth = nn.Sigmoid()
        self.beta = beta
        self.simple = simple # the simple version of ITE evaluates the same function for all dimensions of the output
        super().__init__(submodules, input_type, output_type, input_size, output_size, num_units, name=name)

    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        if self.input_type == 'list':
            assert len(batch.size()) == 3
            assert batch_lens is not None
        else:
            assert len(batch.size()) == 2
        if is_sequential:
            predicted_eval = self.submodules["evalfunction"].execute_on_batch(batch, batch_lens, is_sequential=False)
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens, is_sequential=is_sequential)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens, is_sequential=is_sequential)
        else:
            predicted_eval = self.submodules["evalfunction"].execute_on_batch(batch, batch_lens)
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens)

        gate = self.bsmooth(predicted_eval*self.beta)
        if self.simple:
            gate = gate.repeat(1, self.output_size)
        
        if self.get_typesignature() == ('list', 'list'):
            gate = gate.unsqueeze(1).repeat(1, batch.size(1), 1)
        elif self.get_typesignature() == ('list', 'atom') and is_sequential:
            gate = gate.unsqueeze(1).repeat(1, batch.size(1), 1)

        assert gate.size() == predicted_function2.size() == predicted_function1.size()
        ite_result = gate*predicted_function1 + (1.0 - gate)*predicted_function2

        return ite_result

class SimpleITE(ITE):
    """The simple version of ITE evaluates one function for all dimensions of the output."""

    def __init__(self, input_type, output_type, input_size, output_size, num_units, eval_function=None, function1=None, function2=None, beta=1.0):
        super().__init__(input_type, output_type, input_size, output_size, num_units, 
            eval_function=eval_function, function1=function1, function2=function2, beta=beta, name="SimpleITE", simple=True)
        
class MultiplyFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        if function1 is None:
            function1 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        if function2 is None:
            function2 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        submodules = { "function1" : function1, "function2" : function2 }
        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name="Multiply")

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        predicted_function1 = self.submodules["function1"].execute_on_batch(batch)
        predicted_function2 = self.submodules["function2"].execute_on_batch(batch)
        return predicted_function1 * predicted_function2

class AddFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        if function1 is None:
            function1 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        if function2 is None:
            function2 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name="Add")

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        predicted_function1 = self.submodules["function1"].execute_on_batch(batch)
        predicted_function2 = self.submodules["function2"].execute_on_batch(batch)
        return predicted_function1 + predicted_function2

class ContinueFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, fxn=None):
        if fxn is None:
            fxn = init_neural_function("atom", "atom", input_size, output_size, num_units)
        submodules = { "fxn" : fxn }
        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name="")

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        fxn_out = self.submodules["fxn"].execute_on_batch(batch)
        return fxn_out

class LearnedConstantFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__({}, "atom", "atom", input_size, output_size, num_units, name="LearnedConstant", has_params=True)

    def init_params(self):
        self.parameters = { "constant" : torch.rand(self.output_size, requires_grad=True, device=device) }

    def execute_on_batch(self, batch, batch_lens=None):
        return self.parameters["constant"].unsqueeze(0).repeat(batch.size(0), 1)
        
class AffineFunction(LibraryFunction):

    def __init__(self, raw_input_size, selected_input_size, output_size, num_units, name="Affine"):
        self.selected_input_size = selected_input_size
        super().__init__({}, "atom", "atom", raw_input_size, output_size, num_units, name=name, has_params=True)

    def init_params(self):
        self.linear_layer = nn.Linear(self.selected_input_size, self.output_size, bias=True).to(device)
        self.parameters = {
            "weights" : self.linear_layer.weight,
            "bias" : self.linear_layer.bias
        }

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        return self.linear_layer(batch)

class AffineFeatureSelectionFunction(AffineFunction):

    def __init__(self, input_size, output_size, num_units, name="AffineFeatureSelection"):
        assert hasattr(self, "full_feature_dim")
        assert input_size >= self.full_feature_dim
        if self.full_feature_dim == 0:
            self.is_full = True
            self.full_feature_dim = input_size
        else:
            self.is_full = False
        additional_inputs = input_size - self.full_feature_dim

        assert hasattr(self, "feature_tensor")
        assert len(self.feature_tensor) <= input_size
        self.feature_tensor = self.feature_tensor.to(device)
        super().__init__(raw_input_size=input_size, selected_input_size=self.feature_tensor.size()[-1]+additional_inputs, 
            output_size=output_size, num_units=num_units, name=name)

    def init_params(self):
        self.raw_input_size = self.input_size
        if self.is_full:
            self.full_feature_dim = self.input_size
            self.feature_tensor = torch.arange(self.input_size).to(device)

        additional_inputs = self.raw_input_size - self.full_feature_dim
        self.selected_input_size = self.feature_tensor.size()[-1] + additional_inputs
        self.linear_layer = nn.Linear(self.selected_input_size, self.output_size, bias=True).to(device)
        self.parameters = {
            "weights" : self.linear_layer.weight,
            "bias" : self.linear_layer.bias
        }

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        features = torch.index_select(batch, 1, self.feature_tensor)
        remaining_features = batch[:,self.full_feature_dim:]
        return self.linear_layer(torch.cat([features, remaining_features], dim=-1))

class FullInputAffineFunction(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = 0 # this will indicate additional_inputs = 0 in FeatureSelectionFunction
        self.feature_tensor = torch.arange(input_size) # selects all features by default
        super().__init__(input_size, output_size, num_units, name="FullFeatureSelect")
