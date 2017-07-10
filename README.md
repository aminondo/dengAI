# dengAI Competition

This repo contains different models I built for drivendata's DengAI competition: https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/


#### dengai.ipynb

This is my first model, implemented a tensorflow DNN as well as perform explanatory analysis from the competition's benchmark. Performed really poorly.

#### dengai-mlp.ipynb

This model contains an implementation of a multi layer perceptron. Also didn't perform so good.

#### dengai-xgboost.ipynb

This model contains an implementation of a xgboost tree model. It actually performed quite well scoring a 26 MAE

#### dengai-stacked-model.ipynb

This model blends a tensorflow DNN and an xgboost tree. I also added lagged variables. This model has performed the best so far scoring me a 25.6 MAE.


#### dengai-stacked-model2.ipynb

This model adds rolling averages to previous stacked model. It improved my prediction to 24.7 MAE. Moved me into the top  17 percentile!
