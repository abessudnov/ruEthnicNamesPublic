import re

import numpy as np
import pandas as pd

from transliterate import translit
from transliterate.exceptions import LanguageDetectionError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data/allNamesClean.csv'
POSSIBLE_TRANSLITERATIONS = ['ru', 'en']


# Load data without any preprocessing
def get_raw_data(data_path=DATA_PATH):
    return pd.read_csv(data_path, index_col='id')


'''
Filter given data frame with some properties
    data -- data frame
    use_translit -- whether to apply transliteration or not
    transliteration_to -- target language to transliterate
    use_regex -- whether to filter data with given regular expression or not
    regex -- regular expression to filter, data which does not math it is droped
    confidence_threshold -- lower bound for allowed confidence coefficient
Returns filtered data
'''
def filter_data(data,
                use_translit=True, transliteration_to='ru',
                use_regex=True, regex=r'[а-я\-\ ]*',
                confidence_threshold=0):
    data_filtered = data.copy()

    # Transliterate if needed
    if use_translit:
        if transliteration_to not in POSSIBLE_TRANSLITERATIONS:
            raise ValueError('Wrong transliteration option')
        elif transliteration_to == 'ru':
            ''' 
            Transliterate each first and last name to russian and replace ukrainian 'i' with russian 'и' because 
            `trasnlit` function handles only latin characters
            '''
            data_filtered['first_name'] = data_filtered['first_name'].apply(
                lambda x: translit(x, 'ru').replace('і', 'и').replace('І', 'И'))
            data_filtered['last_name'] = data_filtered['last_name'].apply(
                lambda x: translit(x, 'ru').replace('і', 'и').replace('І', 'И'))
        elif transliteration_to == 'en':
            # Function for transliteration to English
            def try_transliterate(val):
                try:
                    return translit(val, reversed=True)
                except LanguageDetectionError:
                    # If failed to detect source language than leave without changes
                    return val
                except:
                    # If other any problem occurred skip
                    return None

            data_filtered['first_name'] = data_filtered['first_name'].apply(try_transliterate)
            data_filtered['last_name'] = data_filtered['last_name'].apply(try_transliterate)
            data_filtered = data_filtered.dropna()  # Drop invalid values

    # Filter out with given regexp if needed
    if use_regex:
        def regex_filter(val):
            if val:
                mo = re.fullmatch(regex, val, re.IGNORECASE)
                if mo:
                    return True
                else:
                    return False
            else:
                return False

        data_filtered = data_filtered[data_filtered['first_name'].apply(regex_filter)]
        data_filtered = data_filtered[data_filtered['last_name'].apply(regex_filter)]

    # Leave data with big enough confidence
    data_filtered = data_filtered[data_filtered['confidence'] >= confidence_threshold]

    return data_filtered


# Rules for aggregation. Classes which are not listed stay the same
aggregation_mapping = {
    'Bashkir': 'BashTat',
    'Belarusian': 'BelRusUkr',
    'Chechen': 'CheDagIng',
    'Dagestani': 'CheDagIng',
    'Ingush': 'CheDagIng',
    'KabardinAdyghe': 'KabAdKarBalOs',
    'KarachayBalkar': 'KabAdKarBalOs',
    'Kazakh': 'KazKyr',
    'Kyrgyz': 'KazKyr',
    'Ossetian': 'KabAdKarBalOs',
    'Russian': 'BelRusUkr',
    'Tajik': 'TajUzb',
    'Tatar': 'BashTat',
    'Ukrainian': 'BelRusUkr',
    'Uzbek': 'TajUzb',
}


'''
Apply aggregation to the given ethnos
    ethn -- ethnos
Returns aggregated ethnos
'''
def aggregate_ethnos(ethn):
    if ethn not in aggregation_mapping:
        return ethn
    return aggregation_mapping[ethn]


'''
Move intersection of train and test sets to train set
    X_train -- train names set
    X_test -- test names set
    y_train -- train labels
    y_test -- test labels
    X_train_s -- train sexes set
    X_test_s -- test sexes set
Returns modified sets
'''
def fix_data_leak(X_train, X_test, y_train, y_test, X_train_s, X_test_s):
    indices = np.where(np.in1d(X_test, X_train))[0]  # indices in test of elements which occur in both sets

    # Move elements to train sets
    X_train = np.append(X_train, X_test[indices])
    y_train = np.append(y_train, y_test[indices])
    X_train_s = np.append(X_train_s, X_test_s[indices])

    # Remove elements from test sets
    X_test = np.delete(X_test, indices)
    y_test = np.delete(y_test, indices)
    X_test_s = np.delete(X_test_s, indices)

    return X_train, X_test, y_train, y_test, X_train_s, X_test_s


'''
Split data to train and test subsets
    data -- data to split
    test_size -- fraction of test
    random_state -- random seed for splitting
    delimiter -- delimiter to separate first and last name
    fix_leak -- whether to fix data leak or not
    le -- predefined label encoder
Returns label encoder and splitted data with corresponding sex label for each entry      
'''
def split_data_with_sexes(data, test_size=0.25, random_state=0, delimiter='#', fix_leak=True, le=None):
    names = (data.last_name + delimiter + data.first_name).to_numpy()  # Concat first and last name
    sexes = data.sex
    labels = data.ethn

    # Encode ethnicity labels with numbers
    if le is None:
        le = LabelEncoder()
        labels_enc = le.fit_transform(labels)
    else:
        labels_enc = le.transform(labels)

    # Split pairs of (name, sex) and label
    if test_size != 0.0:
        X_train_t, X_test_t, y_train, y_test = train_test_split(list(zip(names, sexes)), labels_enc, test_size=test_size,
                                                                stratify=labels_enc, random_state=random_state)
    else:
        X_train_t = list(zip(names, sexes))
        y_train = labels_enc
        X_test_t = []
        y_test = []

    # Separate name and sex for each entry
    X_train = []
    X_test = []

    X_train_s = []
    X_test_s = []

    for n, s in X_train_t:
        X_train.append(n)
        X_train_s.append(s)
    for n, s in X_test_t:
        X_test.append(n)
        X_test_s.append(s)

    # Converting data to numpy arrays
    X_train, X_test, y_train, y_test, X_train_s, X_test_s = np.array(X_train), np.array(X_test), \
                                                            np.array(y_train), np.array(y_test), \
                                                            np.array(X_train_s), np.array(X_test_s)

    # Data leak fixing
    if fix_leak:
        X_train, X_test, y_train, y_test, X_train_s, X_test_s = fix_data_leak(X_train, X_test, y_train, y_test,
                                                                              X_train_s, X_test_s)
    return le, X_train, X_test, y_train, y_test, X_train_s, X_test_s


'''
Split data to train and test subsets
    data -- data to split
    test_size -- fraction of test
    random_state -- random seed for splitting
    delimiter -- delimiter to separate first and last name
    fix_leak -- whether to fix data leak or not
    le -- predefined label encoder
Returns label encoder and splitted data      
'''
def split_data(data, test_size=0.25, random_state=0, delimiter='#', fix_leak=True, le=None):
    le, X_train, X_test, y_train, y_test, _, _ = split_data_with_sexes(data, test_size,
                                                                       random_state, delimiter, fix_leak, le)
    return le, X_train, X_test, y_train, y_test


'''
Split data to train and test subsets where data is already vectorized
    data -- data to split
    test_size -- fraction of test
    random_state -- random seed for splitting
Returns label encoder and splitted data      
'''
def split_vectorized_data(data, test_size=0.25, random_state=0):
    X = []
    y = []
    # Extract vectors and labels
    for k in data:
        X.append(np.concatenate([data[k][1], data[k][2]]))
        y.append(data[k][-1])

    # Encode ethnicity labels with numbers
    le = LabelEncoder()
    labels_enc = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels_enc, test_size=test_size, stratify=labels_enc,
                                                        random_state=random_state)
    return le, np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


'''
Augment data with random shuffling of first and last names
    X -- original names
    y -- original classes
    sex -- sexes of entries
    upsample_limit -- maximum number of entries for each class
    delimiter -- symbol which separates first and last names 
    random_state -- random seed for choosing pairs to shuffle
Returns augmented data
'''
def upsample(X, y, sex, upsample_limit=6000, delimiter='#', random_seed=0):
    X = list(X)  # new data
    y_n = list(y)  # new labels
    X_a = np.array(X)  # original data
    ethns = np.unique(y)  # original labels

    np.random.seed(random_seed)
    # Process each ethnos independently
    for ethn in ethns:
        # Take only needed ethnos and evaluated number of new entries
        X_e = X_a[y == ethn]
        size = len(X_e)
        up_n = upsample_limit - size

        # Split data corresponding to sex
        X_es = [[], []]
        for i, n in enumerate(list(X_e)):
            if sex[i] == 1:
                X_es[0].append(n)
            else:
                X_es[1].append(n)

        # Obtain new samples by taking random pair of people with same sex and ethnos
        new_samples = []
        while up_n > 0:
            for s in (0, 1):
                ids = np.random.randint(0, len(X_es[s]), size=2)
                p1 = X_es[s][ids[0]].split(delimiter)
                p2 = X_es[s][ids[1]].split(delimiter)
                new_samples.append(p1[0] + delimiter + p2[1])
                new_samples.append(p2[0] + delimiter + p1[1])

                y_n += [ethn] * 2
                up_n -= 2

        X += new_samples

    return X, np.array(y_n)
