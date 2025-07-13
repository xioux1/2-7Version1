import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import pipeline

def test_prepare_matrices_drops_high_corr_taxes():
    train_df = pd.DataFrame({
        'Id': [1, 2, 3, 4],
        'ranker_id': [1, 1, 2, 2],
        'selected': [0, 1, 0, 1],
        'profileId': [0, 0, 0, 0],
        'companyID': [1, 1, 1, 1],
        'searchRoute': ['A', 'A', 'B', 'B'],
        'legs0_segments0_departureFrom_airport_iata': ['SFO', 'LAX', 'SFO', 'LAX'],
        'requestDate': pd.to_datetime(['2024-01-01'] * 4),
        'legs0_departureAt': pd.to_datetime(['2024-01-02'] * 4),
        'legs0_arrivalAt': pd.to_datetime(['2024-01-02 01:00'] * 4),
        'legs1_departureAt': pd.to_datetime(['2024-01-02'] * 4),
        'legs1_arrivalAt': pd.to_datetime(['2024-01-02 03:00'] * 4),
        'taxes': [10.0, 20.0, 30.0, 40.0],
        'totalPrice': [100.0, 200.0, 300.0, 400.0],
        'price_from_median': [10.0, 20.0, 30.0, 40.0],
    })
    test_df = train_df.drop(columns=['selected'])

    X, y, X_test, _ = pipeline.prepare_matrices(train_df.copy(), test_df.copy())

    assert 'taxes' not in X.columns
    assert 'taxes' not in X_test.columns


def test_prepare_matrices_uses_corr_thresh_parameter():
    train_df = pd.DataFrame({
        'Id': [1, 2, 3, 4],
        'ranker_id': [1, 1, 2, 2],
        'selected': [0, 1, 0, 1],
        'profileId': [0, 0, 0, 0],
        'companyID': [1, 1, 1, 1],
        'searchRoute': ['A', 'A', 'B', 'B'],
        'legs0_segments0_departureFrom_airport_iata': ['SFO', 'LAX', 'SFO', 'LAX'],
        'requestDate': pd.to_datetime(['2024-01-01'] * 4),
        'legs0_departureAt': pd.to_datetime(['2024-01-02'] * 4),
        'legs0_arrivalAt': pd.to_datetime(['2024-01-02 01:00'] * 4),
        'legs1_departureAt': pd.to_datetime(['2024-01-02'] * 4),
        'legs1_arrivalAt': pd.to_datetime(['2024-01-02 03:00'] * 4),
        'taxes': [10.0, 20.0, 30.0, 40.0],
        'totalPrice': [100.0, 200.0, 300.0, 400.0],
        'price_from_median': [10.0, 20.0, 30.0, 40.0],
    })
    test_df = train_df.drop(columns=['selected'])

    X, _, X_test, _ = pipeline.prepare_matrices(train_df.copy(), test_df.copy(), corr_thresh=1.0)

    assert 'taxes' in X.columns
    assert 'taxes' in X_test.columns
