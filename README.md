# mylinearsvm

### mylinearsvm.py

* Contains implementation of linear SVM using Huberized-hinge loss and one vs. rest multi-class classification.

### random-demo.py

* Uses a random simulated dataset (all 8 classes' features follow random Normal distributions).
* Outputs training process to the console (classifier, iteration, and objective value)
* Outputs training and test error to the console

### digits-demo.py

* Uses scikit-learn's handwritten digits dataset
* Outputs training process to the console (classifier, iteration, and objective value)
* Outputs training and test error to the console

### skl-comparison.py

* Uses scikit-learn's handwritten digits dataset
* Accepts command line arguments (optimal regularization parameters are the default)
* Outputs selected parameters to the console
* Outputs training and test error to the console

Ex: python skl-comparison.py [--lambda lambda] [--C C] [--intercept]

_lambda_: regularization parameter for my linear SVM (must be numeric)

_C_: regularization parameter for scikit-learn's linear SVM (must be numeric)

_intercept_: if set, fits scikit-learn's linear SVM with the intercept included (it is excluded by default because my linear SVM doesn't fit with an intercept and scikit-learn's implementation has far superior performance when the intercept is included)
