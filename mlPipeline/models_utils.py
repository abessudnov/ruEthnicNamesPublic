import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, \
    accuracy_score, make_scorer

# Set size and font size for plots
sns.set(rc={'figure.figsize': (7, 4.2)})
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
sns.set(font_scale=1.)

f1 = make_scorer(f1_score, average='macro')  # Function to maximize with grid search

DEFAULT_VECT_PARAMS = {
    'analyzer': ['char'],
    'ngram_range': [(1, 1), (1, 3), (1, 5), (1, 7)],
    'max_df': [0.05, 0.1, 0.2],
    'min_df': [1, 10, 0.01, 0.05],
    'lowercase': [False, True],
}

AVAILABLE_VECTORIZATION = ['tfidf', 'count', None]


'''
Perform grid search with given pipeline and parameters
    model -- estimator to optimize
    model_params -- search space for estimator parameters
    X_train -- train data
    y_train -- train labels
    vect_params -- search space for vectorization parameters
    vectorization -- vectorization type to perform before fitting data into estimator
    n_jobs -- number of CPU threads to use
    cv -- number of cross-validation folds 
Returns grid search result
'''
def try_model(model, model_params, X_train, y_train,
              vect_params=None, vectorization='tfidf',
              n_jobs=6, cv=3):
    pipe = None  # Full pipeline
    pipe_params = {}  # Pipeline params
    # Process vectorization
    if vectorization is not None:
        if vect_params is None:
            vect_params = DEFAULT_VECT_PARAMS
        vectorizer = None

        if vectorization not in AVAILABLE_VECTORIZATION:
            raise ValueError('Wrong vectorization option, supported are:', AVAILABLE_VECTORIZATION)
        elif vectorization == 'tfidf':
            vectorizer = TfidfVectorizer()
        elif vectorization == 'count':
            vectorizer = CountVectorizer()

        pipe = Pipeline(steps=[('vect', vectorizer), ('model', model)])  # Build pipeline

        # Merge estimator and vectorization params
        for p in vect_params:
            pipe_params['vect__' + p] = vect_params[p]
        for p in model_params:
            pipe_params['model__' + p] = model_params[p]
    else:
        pipe = model
        pipe_params = model_params

    # Run grid search with f1 marco score as function for maximization on cross-validation
    clf = GridSearchCV(pipe, pipe_params, verbose=3, n_jobs=n_jobs, cv=cv, scoring=f1)
    clf.fit(X_train, y_train)

    return clf


'''
Estimate metrics for each class independently
    pred -- predicted labels
    y -- true labels
    le -- label encoder
'''
def evaluate_classes(pred, y, le):
    print('Ethnos'.ljust(15), end='')
    print(' Acc\t', end='')
    print('Prec\t', end='')
    print('Rec\t', end='')
    print('F1')
    for i in range(len(le.classes_)):
        # inds = y == i  # indices with true i-th ethnos
        # y_c = (y[inds] == i).astype(int)
        # pred_c = (pred[inds] == i).astype(int)
        inds2 = (y == i) | (pred == i)  # indices with true or predicted i-th ethnos
        y_c2 = (y[inds2] == i).astype(int)
        pred_c2 = (pred[inds2] == i).astype(int)
        # Estimate metrics and print them
        print(le.inverse_transform([i])[0].ljust(15), end='\t')
        print("%1.4f" % accuracy_score((y == i).astype(int), (pred == i).astype(int)), end='\t')
        print("%1.4f" % precision_score(y_c2, pred_c2), end='\t')
        print("%1.4f" % recall_score(y_c2, pred_c2), end='\t')
        print("%1.4f" % f1_score(y_c2, pred_c2))

'''
Estimate metrics for given data and draw confusion matrix
    model -- model to estimate
    le -- label encoder
    X -- input data
    y -- labels
    hm_save_path -- path to save heat map if not None
Returns metrics
'''
def test_model(model, le, X, y, hm_save_path=None):
    pred = model.predict(X)

    # Draw confusion matrix
    cm = confusion_matrix(y, pred, normalize='true')
    cm = np.around(cm, 2)
    ax = sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_,
                     annot_kws={"fontsize": 8}, cmap="YlGnBu", cbar=False)
    #ax.set_xlabel("Predicted", fontsize=8)
    #ax.set_ylabel("Actual", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.tick_params(axis='both', which='both', length=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if hm_save_path is not None:
        plt.savefig(hm_save_path, bbox_inches='tight')
    plt.show()

    # Estimate metrics for each class independently
    evaluate_classes(pred, y, le)

    # Estimate metrics for model
    return {
        'accuracy': accuracy_score(y, pred),
        'precision macro': precision_score(y, pred, average='macro'),
        'recall macro': recall_score(y, pred, average='macro'),
        'f1 macro': f1_score(y, pred, average='macro'),
    }


'''
Finds best estimator with grid search and tests it
    model -- estimator to optimize
    model_params -- search space for estimator parameters
    X_train -- train data
    y_train -- train labels
    X_test -- test data
    y_test -- test labels
    vect_params -- search space for vectorization parameters
    vectorization -- vectorization type to perform before fitting data into estimator
    n_jobs -- number of CPU threads to use
    cv -- number of cross-validation folds
Returns best estimator 
'''
def proc_model(estimator, params, le, X_train, y_train, X_test, y_test,
               vect_params=None, vectorization='tfidf', cv=3, n_jobs=6):
    if vect_params is None:
        vect_params = DEFAULT_VECT_PARAMS

    # Run grid search
    grid = try_model(estimator, params, X_train, y_train, vect_params, vectorization, cv=cv, n_jobs=n_jobs)
    print(grid.best_params_)

    # Take best model and run it on test data
    best_model = grid.best_estimator_
    results = test_model(best_model, le, X_test, y_test)
    print(results)

    return best_model
