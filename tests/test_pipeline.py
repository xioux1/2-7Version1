import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from unittest.mock import patch
import pipeline


def test_pipeline_main_returns_expected_columns():
    # Dummy data resembling required columns
    train_df = pd.DataFrame({
        'Id': [1, 2],
        'ranker_id': [1, 1],
        'selected': [1, 2],
        'feature': [0.1, 0.2]
    })

    test_df = pd.DataFrame({
        'Id': [3, 4],
        'ranker_id': [1, 2],
        'feature': [0.3, 0.4]
    })

    sample_submission_df = pd.DataFrame({'Id': [3, 4], 'ranker_id': [1, 2], 'selected': [0, 0]})
    test_ids_df = test_df[['Id', 'ranker_id']]

    # Mock pipeline functions to simplify processing
    with patch('pipeline.load_data', return_value=(train_df, test_df, sample_submission_df, test_ids_df)), \
         patch('pipeline.preprocess_dataframe', side_effect=lambda df, is_train: df), \
         patch('pipeline.prepare_matrices', side_effect=lambda a, b: (a[["feature"]], a['selected'], b[["feature"]], a['ranker_id'])), \
         patch('pipeline.encode_categoricals', side_effect=lambda X, X_test: (X, X_test, [])), \
         patch('pipeline.clean_features', side_effect=lambda X, X_test, low_var_thresh=1: (X, X_test, {})), \
         patch('lightgbm.LGBMRanker.fit', return_value=None), \
         patch('pandas.DataFrame.to_parquet', return_value=None), \
         patch('pandas.DataFrame.to_csv', return_value=None):

        def dummy_train_model(X, y, X_test, ranker_ids, cat_features):
            # Call fit once to satisfy mocked fit
            model = pipeline.lgb.LGBMRanker()
            model.fit(X, y, group=[len(X)])
            preds = np.zeros(len(X_test))
            return preds, None

        with patch('pipeline.train_model', side_effect=dummy_train_model):
            result = pipeline.main()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['Id', 'ranker_id', 'selected']
