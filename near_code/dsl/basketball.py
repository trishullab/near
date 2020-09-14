import torch
from .library_functions import AffineFeatureSelectionFunction


DEFAULT_BBALL_FEATURE_SUBSETS = {
    "ball"      : torch.LongTensor([0, 1]),
    "offense"   : torch.LongTensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    "defense"   : torch.LongTensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
}
BBALL_FULL_FEATURE_DIM = 22


class BBallBallSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["ball"]
        super().__init__(input_size, output_size, num_units, name="BallXYAffine")

class BBallOffenseSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["offense"]
        super().__init__(input_size, output_size, num_units, name="OffenseXYAffine")

class BBallDefenseSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["defense"]
        super().__init__(input_size, output_size, num_units, name="DefenseXYAffine")
