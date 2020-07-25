import attr


@attr.s
class DarkNetPredictionResult:
    class_name = attr.ib(type=str)  # name of the class detected. E.g. dog.
    class_confidence = attr.ib(type=float)  # probability of the class. E.g. 95%.
    left_x = attr.ib(type=int)  # box center x.
    top_y = attr.ib(type=int)  # box center y.
    width = attr.ib(type=int)  # box width.
    height = attr.ib(type=int)  # box height.
    info = attr.ib(type=dict, default={})  # extra info. Can be blank.
