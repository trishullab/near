import dsl

DSL_DICT = {('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
                        ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                                            dsl.running_averages.RunningAverageLast10Function, dsl.running_averages.RunningAverageWindow11Function,
                                            dsl.running_averages.RunningAverageWindow5Function],
                        ('atom', 'atom') : [dsl.SimpleITE, dsl.AddFunction, dsl.MultiplyFunction, dsl.crim13.Crim13PositionSelection, 
                                            dsl.crim13.Crim13DistanceSelection, dsl.crim13.Crim13DistanceChangeSelection,
                                            dsl.crim13.Crim13VelocitySelection, dsl.crim13.Crim13AccelerationSelection,
                                            dsl.crim13.Crim13AngleSelection, dsl.crim13.Crim13AngleChangeSelection]}

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}

