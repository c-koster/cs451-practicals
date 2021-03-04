# Decision Trees: Feature Splits

#%%
# Python typing introduced in 3.5: https://docs.python.org/3/library/typing.html
from typing import List

# As of Python 3.7, this exists! https://www.python.org/dev/peps/pep-0557/
from dataclasses import dataclass

# My python file (very limited for now, but we will build up shared functions)
from shared import TODO

#%%
# Let's define a really simple class with two fields:
@dataclass
class DataPoint:
    temperature: float
    frozen: bool

    def get_temp(self) -> float:
        return self.temperature


# Fahrenheit, sorry.
data = [
    # vermont temperatures; frozen=True
    DataPoint(0, True),
    DataPoint(-2, True),
    DataPoint(10, True),
    DataPoint(11, True),
    DataPoint(6, True),
    DataPoint(28, True),
    DataPoint(31, True),
    # warm temperatures; frozen=False
    DataPoint(33, False),
    DataPoint(45, False),
    DataPoint(76, False),
    DataPoint(60, False),
    DataPoint(34, False),
    DataPoint(98.6, False),
]


def is_water_frozen(temperature: float) -> bool:
    """
    This is how we **should** implement it.
    """
    return temperature <= 32


# Make sure the data I invented is actually correct...
for d in data:
    assert d.frozen == is_water_frozen(d.temperature)


def find_candidate_splits(data: List[DataPoint]) -> List[float]:
    data.sort(key=lambda pt: pt.get_temp())

    midpoints = []
    for i in range(len(data)-1):
        l = data[i].get_temp()
        r = data[i+1].get_temp()

        mid = (l + r) / 2 #calculate the midpoint
        midpoints.append(mid)

    return midpoints


def gini_impurity(points: List[DataPoint]) -> float:
    """
    The standard version of gini impurity sums over the classes:
    """
    p_ice = sum(1 for x in points if x.frozen) / len(points)
    p_water = 1.0 - p_ice
    return p_ice * (1 - p_ice) + p_water * (1 - p_water)
    # for binary gini-impurity (just two classes) we can simplify, because 1 - p_ice == p_water, etc.
    # p_ice * p_water + p_water * p_ice
    # 2 * p_ice * p_water
    # not really a huge difference.


def impurity_of_split(points: List[DataPoint], split: float) -> float:
    smaller = []
    bigger = []
    for p in points: # loop over all my points
        if p.temperature > split:
            bigger.append(p)
        else:
            smaller.append(p)
    # get used to putting assert statements everywhere
    assert( (len(smaller)+len(bigger)) == len(points))

    return gini_impurity(smaller) + gini_impurity(bigger)


if __name__ == "__main__":
    print("Initial Impurity: ", gini_impurity(data))
    print("Impurity of first-six (all True): ", gini_impurity(data[:6]))
    print("")
    for split in find_candidate_splits(data):
        score = impurity_of_split(data, split)
        print("splitting at {} gives us impurity {}".format(split, score))
        if score == 0.0:
            break
