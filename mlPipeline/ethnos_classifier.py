import re
import warnings

import pandas as pd
from transliterate import translit

from pkl_utils import load_pkl


def filter_data(data):
    regex = r'[а-я\-\ ]*'  # Available characters
    data_filtered = data.copy()

    data_filtered['first_name'] = data_filtered['first_name'].apply(str)
    data_filtered['last_name'] = data_filtered['last_name'].apply(str)
    ''' 
    Transliterate each first and last name to russian and replace ukrainian 'i' with russian 'и' because 
    `trasnlit` function handles only latin characters
    '''
    data_filtered['first_name'] = data_filtered['first_name'].apply(
        lambda x: translit(x, 'ru').replace('і', 'и').replace('І', 'И'))
    data_filtered['last_name'] = data_filtered['last_name'].apply(
        lambda x: translit(x, 'ru').replace('і', 'и').replace('І', 'И'))

    # Filter out with given regexp, invalid names are replaced with empty strings
    def regex_filter(val):
        if val:
            mo = re.fullmatch(regex, val, re.IGNORECASE)
            if mo:
                return val
            else:
                return ''
        else:
            return ''

    data_filtered['first_name'] = data_filtered['first_name'].apply(regex_filter)
    data_filtered['last_name'] = data_filtered['last_name'].apply(regex_filter)

    return data_filtered


def get_class_prediction(model, le, names):
    pred = model.predict(names).astype(int)
    pred = le.inverse_transform(pred)
    pred[names == '#'] = 'INVALID'
    return pred


class EthnosClassifier:
    def __init__(self):
        self.model = load_pkl('models/all_classes_model.pkl')
        self.model_agr = load_pkl('models/aggregated_classes_model.pkl')
        self.le = load_pkl('models/label_encoder.pkl')
        self.le_agr = load_pkl('models/aggregated_label_encoder.pkl')

    def predict(self, csv_path, dest_path, classification='both', predict_probs=False):
        if classification not in ['both', 'full', 'aggregated']:
            raise ValueError('Classification mode must be "full" or "aggregated"')

        # Load and validate data
        df = pd.read_csv(csv_path)
        if 'first_name' not in df.columns or 'last_name' not in df.columns:
            raise IndexError('CSV file must contain "first_name" and "last_name" columns')

        filtered_df = filter_data(df)

        # Concat first and last names with delimiter to pass to the classifier
        names = (filtered_df.last_name + '#' + filtered_df.first_name).to_numpy()

        if '#' in names:
            warnings.warn(
                'Warning: Some names in your data contain characters different from russian letters, "-" character or '
                'space. Results for this names will be "INVALID" or 0.0 for probabilities')

        res_df = df[['first_name', 'last_name']]

        if classification in ['full', 'both']:
            if not predict_probs:
                res_df.loc[:, 'ethnos'] = get_class_prediction(self.model, self.le, names)
            else:
                probs = self.model.predict_proba(names)
                for i, cls in enumerate(self.le.classes_):
                    pr = probs[:, i]
                    pr[names == '#'] = 0.0
                    res_df.loc[:, cls + '_full'] = pr

        if classification in ['both', 'aggregated']:
            if not predict_probs:
                res_df.loc[:, 'agr_ethnos'] = get_class_prediction(self.model_agr, self.le_agr, names)
            else:
                probs = self.model_agr.predict_proba(names)
                for i, cls in enumerate(self.le_agr.classes_):
                    pr = probs[:, i]
                    pr[names == '#'] = 0.0
                    res_df.loc[:, cls + '_agr'] = pr

        res_df.to_csv(dest_path, index=False)
