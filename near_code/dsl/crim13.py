import torch
from .library_functions import AffineFeatureSelectionFunction


CRIM13_FEATURE_SUBSETS = {
    "position" : torch.LongTensor([0, 1, 2, 3]),
    "distance" : torch.LongTensor([4]), 
    "distance_change" : torch.LongTensor([5]), 
    "velocity" : torch.LongTensor([11, 12, 13, 14]),
    "acceleration" : torch.LongTensor([15, 16, 17, 18]),
    "angle" : torch.LongTensor([6, 7, 10]),    
    "angle_change" : torch.LongTensor([8, 9])    
}
CRIM13_FULL_FEATURE_DIM = 19


class Crim13PositionSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = CRIM13_FULL_FEATURE_DIM        
        self.feature_tensor = CRIM13_FEATURE_SUBSETS["position"]
        super().__init__(input_size, output_size, num_units, name="PositionSelect")

class Crim13DistanceSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = CRIM13_FULL_FEATURE_DIM
        self.feature_tensor = CRIM13_FEATURE_SUBSETS["distance"]
        super().__init__(input_size, output_size, num_units, name="DistanceSelect")

class Crim13DistanceChangeSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = CRIM13_FULL_FEATURE_DIM
        self.feature_tensor = CRIM13_FEATURE_SUBSETS["distance_change"]
        super().__init__(input_size, output_size, num_units, name="DistanceChangeSelect")

class Crim13VelocitySelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = CRIM13_FULL_FEATURE_DIM
        self.feature_tensor = CRIM13_FEATURE_SUBSETS["velocity"]
        super().__init__(input_size, output_size, num_units, name="VelocitySelect")

class Crim13AccelerationSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = CRIM13_FULL_FEATURE_DIM
        self.feature_tensor = CRIM13_FEATURE_SUBSETS["acceleration"]
        super().__init__(input_size, output_size, num_units, name="AccelerationSelect")

class Crim13AngleSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = CRIM13_FULL_FEATURE_DIM
        self.feature_tensor = CRIM13_FEATURE_SUBSETS["angle"]
        super().__init__(input_size, output_size, num_units, name="AngleSelect")

class Crim13AngleChangeSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = CRIM13_FULL_FEATURE_DIM
        self.feature_tensor = CRIM13_FEATURE_SUBSETS["angle_change"]
        super().__init__(input_size, output_size, num_units, name="AngleChangeSelect")
