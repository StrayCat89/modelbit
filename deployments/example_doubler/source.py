import modelbit, sys
from typing import *
from sklearn.linear_model._base import LinearRegression

lm = modelbit.load_value("data/lm.pkl") # LinearRegression()

# main function
def example_doubler(half: int) -> int:
    if type(half) is not int:
        return None
    return round(lm.predict([[half]])[0])

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = example_doubler(...)
#   print(result)