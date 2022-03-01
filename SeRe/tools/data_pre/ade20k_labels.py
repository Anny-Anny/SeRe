from collections import namedtuple
import scipy.io


Label = namedtuple('Label', [
    'name',
    'id',
    'color'
])

labels = [Label('surfaces', 0, (255, 255, 255)),
          Label('building', 1, (0, 0, 255)),
          Label('low vegetation', 2, (0, 255, 255)),
          Label('Tree', 3, (0, 255, 0)),
          Label('Car', 4, (255, 255, 0)),
          Label('Clutter / background', 5, (255, 0, 0))]
ade20k_id2label = {label.id: label for label in labels}