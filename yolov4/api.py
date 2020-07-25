import attr


@attr.s
class DarkNetPredictionResult:
    # https://www.ccoderun.ca/programming/2019-08-18_Darknet_training_images/
    class_name = attr.ib(type=str)
    class_confidence = attr.ib(type=float)
    left_x = attr.ib(type=int)  # box center x.
    top_y = attr.ib(type=int)  # box center y.
    width = attr.ib(type=int)  # box width.
    height = attr.ib(type=int)  # box height.
    info = attr.ib(type=dict, default={})  # extra info about filename...
