import numpy as np

from collections import namedtuple

FullState = namedtuple('position', 'obstacles', 'litter')

SidewalkState = namedtuple('position')
ForwardState = namedtuple('position')
ObstaclesState = namedtuple('position', 'obstacles')
LitterState = namedtuple('position', 'litter')


