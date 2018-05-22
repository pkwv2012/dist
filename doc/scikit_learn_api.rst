Scikit-learn api Guide
^^^^^^^^^^^^^^^^^^^^^^^

xLearn can support scikit-learn-like api for users. Here is an example: ::

  import numpy as np
  import xlearn as xl
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split

  # Load dataset
  iris_data = load_iris()
  X = iris_data['data']
  y = (iris_data['target'] == 2)

  X_train,   \
  X_val,     \
  y_train,   \
  y_val = train_test_split(X, y, test_size=0.3, random_state=0)

  # param:
  #  0. binary classification
  #  1. model scale: 0.1
  #  2. epoch number: 10 (auto early-stop)
  #  3. learning rate: 0.1
  #  4. regular lambda: 1.0
  #  5. use sgd optimization method
  linear_model = xl.LRModel(task='binary', init=0.1, 
                            epoch=10, lr=0.1, 
                            reg_lambda=1.0, opt='sgd')

  # Start to train
  linear_model.fit(X_train, y_train, 
                   eval_set=[X_val, y_val], 
                   is_lock_free=False)

  # Generate predictions
  y_pred = linear_model.predict(X_val)

In this example, we use linear model to train a binary classifier. We can also 
create FM and FFM by using ``xl.FMModel()`` and ``xl.FMModel()`` . Please see 
the details of these examples in (`Link`__)

.. __: https://github.com/aksnzhy/xlearn/tree/master/demo/classification/scikit_learn_demo
