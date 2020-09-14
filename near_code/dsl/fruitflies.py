import torch
from .library_functions import AffineFeatureSelectionFunction


FRUITFLY_FEATURE_SUBSETS = {
    "linear" : torch.LongTensor([17, 25]),
    "angular" : torch.LongTensor([18, 26, 27]),
    "positional" : torch.LongTensor([24, 28]),
    "ratio" : torch.LongTensor([22, 23]),
    "wing" : torch.LongTensor([19, 20, 21]),
    "FlyStaticFeatures" : torch.LongTensor([0,1,3,4,5,6,8,13,14,15,16,19,20]),
    "FlyDynamicFeatures" : torch.LongTensor([17,18,22,23,24,27]),
    "FlyRelativeFeatures": torch.LongTensor([25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 46])
}
FRUITFLY_FULL_FEATURE_DIM = 53


class FruitFlyWingSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = FRUITFLY_FULL_FEATURE_DIM
        self.feature_tensor = FRUITFLY_FEATURE_SUBSETS["wing"]
        super().__init__(input_size, output_size, num_units, name="WingSelect")

class FruitFlyRatioSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = FRUITFLY_FULL_FEATURE_DIM
        self.feature_tensor = FRUITFLY_FEATURE_SUBSETS["ratio"]
        super().__init__(input_size, output_size, num_units, name="RatioSelect")

class FruitFlyPositionalSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = FRUITFLY_FULL_FEATURE_DIM
        self.feature_tensor = FRUITFLY_FEATURE_SUBSETS["positional"]
        super().__init__(input_size, output_size, num_units, name="PositionalSelect")

class FruitFlyAngularSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = FRUITFLY_FULL_FEATURE_DIM
        self.feature_tensor = FRUITFLY_FEATURE_SUBSETS["angular"]
        super().__init__(input_size, output_size, num_units, name="AngularSelect")

class FruitFlyLinearSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = FRUITFLY_FULL_FEATURE_DIM
        self.feature_tensor = FRUITFLY_FEATURE_SUBSETS["linear"]
        super().__init__(input_size, output_size, num_units, name="LinearSelect")
