import numpy as np

# users: array tipo

# [[Worker, Student, Retired],
# [Male, Female, Unknown]]


# disagg: array tipo

{[0.75, 0.2, 0.05],
 [0.25, 0.30, 0.45]}

# prob: array tipo
# first split
{[
    # worker
    [0.0312, 0.0224, 0.014, 0.0068, 0.0016],
    # student
    [0.0245, 0.0157, 0.0112, 0.005, 0.0007],
    # retired
    [0.0312, 0.0224, 0.014, 0.0068, 0.0016]
],
    # second split
    [
        # male
        [0.0312, 0.0224, 0.014, 0.0068, 0.0016],
        # female
        [0.0312, 0.0224, 0.014, 0.0068, 0.0016],
        # unknown
        [0.0312, 0.0224, 0.014, 0.0068, 0.0016]
    ]}


class ContextGen:
    def __init__(self, users):
        self.users = users
