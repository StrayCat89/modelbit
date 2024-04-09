import modelbit, sys
from typing import *
from lightfm.lightfm import LightFM

model = modelbit.load_value("data/model.pkl") # <lightfm.lightfm.LightFM object at 0x7f3e89e415a0>

# main function
def predict_lightfm(params):
    return model.predict(params["user_ids"], params["feature_ids"])

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = predict_lightfm(...)
#   print(result)