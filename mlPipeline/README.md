# Classifier usage
1. Run `pip install -r requirements.txt`
2. In your Python script import `EthnosClassifier` from `ethnos_classifier.py` and create object `EthnosClassifier`
3. To make predictions call `predict` method of created object. Arguments description:
    1. `csv_path` -- path to file .csv file with names data. It must contain `first_name` and `last_names` columns
    2. `dest_path` -- path to file .csv file to save results
    3. `classification` (optional), possible values:
        1. `both` (default) to make predictions with both aggregated and unaggregated ethnic schemes.
        2. `full` to make predictions only for the unaggregated ethnic scheme.
        3. `aggregated` to make predictions only for the aggregated ethnic scheme.
    4. `predict_probs` (optional) -- boolean value whether to predict the probability for each class or to predict only class, default False
   
<i>See `Classifier_usage.ipynb` for an example of the usage</i>.
# Files description

## 'models' folder
 
- `all_classes_model.pkl`, `aggregated_classes_model.pkl`: sklearn pipline with the best vectorizer and ML algorithm
- `label_encoder.pkl`, `aggregated_label_encoder.pkl`: sklearn LabelEncoder for classification

## Python files

- `data_utils.py`: utilities for data preprocessing
- `ethnos_classifier.py`: the code for the classifier that can be used for new predictions
- `models_utils.py`: utilities for model training, selecting and evaluating
- `nn_utils.py`: MLP utilities
- `pkl_utils.py`: pickle utilities

## Jupyter notebooks

- `Aggragated`: models selecting with the aggregated ethnic scheme
- `Classifier_usage`: example of using the classifier for new predictions
- `FastText_vect` : experiments with FastText vectorisation
- `Final_models`: training, evaluating and saving best model
- `Memo_test`: best model evaluation on the Memorial data
- `Model_selectin_no_leak`: models selection with the unaggregated ethnic scheme
- `Random_classifier`: checking performance of the random classifier
- `Random_users`: selecting random users from Moscow and Kazan and making predictions for them