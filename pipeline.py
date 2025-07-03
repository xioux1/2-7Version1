import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import gc

from utils import (
    create_initial_datetime_features,
    create_remaining_features,
    create_features,
    unify_nan_strategy,
    reduce_mem_usage,
    clean_features,
    calculate_hit_rate_at_3,
    lgb_hit_rate_at_3,
    log_mem_usage,
)

# --- columnas que quiero ignorar en TODO el pipeline ---
COL_BLACKLIST = {
    'legs0_segments1_duration',
}

initial_core_columns = [
    'Id', 'ranker_id', 'selected', 'profileId', 'companyID',
    'requestDate', 'totalPrice', 'taxes',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration',
    'legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_cabinClass',
    'legs0_segments0_baggageAllowance_quantity',
    'searchRoute',
    'pricingInfo_isAccessTP', 'pricingInfo_passengerCount',
    'sex', 'nationality', 'isVip',
    'miniRules0_monetaryAmount', 'miniRules0_percentage',
    'miniRules1_monetaryAmount', 'miniRules1_percentage'
]

extra_cols = [
    'frequentFlyer', 'corporateTariffCode',
    'legs1_segments0_baggageAllowance_quantity',
    'legs1_segments0_cabinClass',
    'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_arrivalTo_airport_iata',
    'legs1_segments0_marketingCarrier_code',
]

initial_core_columns += [c for c in extra_cols if c not in initial_core_columns]
initial_core_columns = [c for c in initial_core_columns if c not in COL_BLACKLIST]
initial_core_columns_test = [c for c in initial_core_columns if c != 'selected']


def load_data(sample_frac=0.5, random_seed=42):
    print("Loading a subset of columns for train_df...")
    train_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet', columns=initial_core_columns)
    log_mem_usage(train_df, "train_df loaded")

    unique_ids = train_df['ranker_id'].unique()
    n_keep = int(len(unique_ids) * sample_frac)
    rng = np.random.RandomState(random_seed)
    sampled_rankers = rng.choice(unique_ids, size=n_keep, replace=False)
    train_df = train_df[train_df['ranker_id'].isin(sampled_rankers)].reset_index(drop=True)
    print(f"Train reducido (grupos completos): {train_df.shape}")
    log_mem_usage(train_df, "train_df sampled")
    bad_groups = train_df.groupby('ranker_id').size().lt(11).sum()
    print(f"Grupos con <11 filas: {bad_groups}")

    print("Loading a subset of columns for test_df...")
    test_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet', columns=initial_core_columns_test)
    log_mem_usage(test_df, "test_df loaded")
    sample_submission_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet')

    if 'Id' in test_df.columns and 'ranker_id' in test_df.columns:
        test_ids_df = test_df[['Id', 'ranker_id']]
    else:
        print("Warning: 'Id' or 'ranker_id' not found in loaded test_df columns. Submission might fail.")
        try:
            test_ids_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet', columns=['Id', 'ranker_id'])
        except Exception:
            test_ids_df = pd.DataFrame()
    return train_df, test_df, sample_submission_df, test_ids_df

def preprocess_dataframe(df, is_train=True):
    df = create_initial_datetime_features(df)
    log_mem_usage(df, "after initial datetime")
    obj_cols = df.select_dtypes('object').columns
    for c in obj_cols:
        df[c] = df[c].astype('category')
    if 'frequentFlyer' in df.columns and pd.api.types.is_categorical_dtype(df['frequentFlyer']):
        df['frequentFlyer'] = df['frequentFlyer'].cat.add_categories(['']).fillna('')
    df = reduce_mem_usage(df)
    log_mem_usage(df, "after reduce_mem_usage 1")
    df = create_features(df)
    log_mem_usage(df, "after create_features")
    df = create_remaining_features(df, is_train=is_train)
    log_mem_usage(df, "after create_remaining_features")
    df = unify_nan_strategy(df)
    log_mem_usage(df, "after unify_nan_strategy")
    df = reduce_mem_usage(df)
    log_mem_usage(df, "after reduce_mem_usage 2")
    return df

def prepare_matrices(train_df_processed, test_df_processed):
    TOP_N = 10
    airport_col = 'legs0_segments0_departureFrom_airport_iata'
    freq = train_df_processed[airport_col].value_counts()
    for df in (train_df_processed, test_df_processed):
        df['dep_freq'] = df[airport_col].map(freq).fillna(1).astype('int32')
    top_hubs = freq.head(TOP_N).index
    for df in (train_df_processed, test_df_processed):
        df['dep_is_hub'] = df[airport_col].isin(top_hubs).astype('int8')

    raw_datetime_col_names = [
        'requestDate', 'legs0_departureAt', 'legs0_arrivalAt',
        'legs1_departureAt', 'legs1_arrivalAt'
    ]
    id_cols_and_target = ['Id', 'ranker_id', 'selected', 'profileId', 'companyID', 'searchRoute']
    DROP_COLS = [
        'free_cancel', 'free_exchange',
        'ff_SU', 'ff_S7', 'ff_U6', 'ff_TK',
        'has_fees', 'has_baggage',
        'price_rank', 'totalPrice_rank_in_group', 'price_from_median',
        'tax_percentage',
        'n_segments_leg0', 'n_segments_leg1',
        'dep_is_hub',
        'group_size_log', 'has_access_tp',
        'is_round_trip',
    ]
    train_df_processed.drop(
        columns=[c for c in DROP_COLS if c in train_df_processed.columns],
        inplace=True,
        errors="ignore",
    )
    test_df_processed.drop(
        columns=[c for c in DROP_COLS if c in test_df_processed.columns],
        inplace=True,
        errors="ignore",
    )

    excluded_for_X_train = id_cols_and_target + raw_datetime_col_names
    train_feature_cols = [c for c in train_df_processed.columns if c not in excluded_for_X_train]
    X = train_df_processed[train_feature_cols].reset_index(drop=True)
    y = train_df_processed['selected'].reset_index(drop=True)
    train_ranker_ids = train_df_processed['ranker_id'].reset_index(drop=True)
    X_test = test_df_processed.reindex(columns=train_feature_cols)

    num_cols_test = X_test.select_dtypes(exclude="category").columns
    X_test[num_cols_test] = X_test[num_cols_test].replace([np.inf, -np.inf], np.nan)

    num_cols_train = X.select_dtypes(exclude="category").columns
    X[num_cols_train] = X[num_cols_train].replace([np.inf, -np.inf], np.nan)
    cat_cols = X_test.select_dtypes("category").columns
    for c in cat_cols:
        if "missing" not in X_test[c].cat.categories:
            X_test[c] = X_test[c].cat.add_categories(["missing"])
        X_test[c] = X_test[c].fillna("missing")
    return X, y, X_test, train_ranker_ids

def encode_categoricals(X, X_test):
    categorical_features_for_encoding = []
    for col in X.columns:
        if X[col].dtype.name in ('object', 'category'):
            categorical_features_for_encoding.append(col)
            le = LabelEncoder()
            if col in X_test.columns:
                combined = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0).unique()
                le.fit(combined)
                X[col] = le.transform(X[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
            else:
                X[col] = le.fit_transform(X[col].astype(str))
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)
            if col in X_test.columns:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1)
    return X, X_test, categorical_features_for_encoding

def train_model(X, y, X_test, ranker_ids, cat_features):
    params = {
        'objective': 'lambdarank',
        'metric': 'None',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'max_bin': 31,
        'lambda_l1':1.0,
        'lambda_l2':1.0,
        'min_gain_to_split': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'importance_type': 'gain',
        'verbose': -1,
        'seed': 42
    }
    NFOLDS = 5
    group_kfold = GroupKFold(n_splits=NFOLDS)
    oof_preds_scores = np.zeros(len(X))
    test_preds_scores = np.zeros(len(X_test))
    fold_hit_rates = []
    feature_importances = pd.DataFrame({'feature': X.columns})

    cat_indices = [X.columns.get_loc(c) for c in cat_features if c in X.columns]
    for fold_, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=ranker_ids)):
        print(f"====== Fold {fold_+1}/{NFOLDS} ======")
        if fold_ > 0:
            gc.collect()
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        train_groups = pd.DataFrame({'ranker_id': ranker_ids.iloc[train_idx]}).groupby('ranker_id', sort=False).size().to_list()
        val_groups = pd.DataFrame({'ranker_id': ranker_ids.iloc[val_idx]}).groupby('ranker_id', sort=False).size().to_list()
        ranker = lgb.LGBMRanker(**params)
        ranker.fit(
            X_train_fold,
            y_train_fold,
            group=train_groups,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_group=[val_groups],
            eval_metric=lgb_hit_rate_at_3,
            callbacks=[lgb.early_stopping(10, verbose=False)],
            categorical_feature=cat_indices if cat_indices else 'auto',
        )
        val_scores = ranker.predict(X_val_fold)
        oof_preds_scores[val_idx] = val_scores
        test_preds_scores += ranker.predict(X_test) / NFOLDS
        val_df = pd.DataFrame({'ranker_id': ranker_ids.iloc[val_idx], 'selected': y_val_fold, 'score': val_scores})
        val_df['predicted_rank'] = val_df.groupby('ranker_id')['score'].rank(method='first', ascending=False).astype(int)
        fold_hr3 = calculate_hit_rate_at_3(val_df)
        fold_hit_rates.append(fold_hr3)
        feature_importances[f'fold_{fold_+1}'] = ranker.booster_.feature_importance(importance_type='gain')
        print(f"Fold {fold_+1} HitRate@3: {fold_hr3:.4f}")
        del ranker, X_train_fold, y_train_fold, X_val_fold, y_val_fold, val_df
        gc.collect()
    oof_df = pd.DataFrame({'ranker_id': ranker_ids, 'selected': y, 'score': oof_preds_scores})
    oof_df['predicted_rank'] = oof_df.groupby('ranker_id')['score'].rank(method='first', ascending=False).astype(int)
    overall_oof_hr3 = calculate_hit_rate_at_3(oof_df)
    print(f"\nOverall OOF HitRate@3: {overall_oof_hr3:.4f}")
    if fold_hit_rates:
        print(f"Mean Fold HitRate@3: {np.mean(fold_hit_rates):.4f}")

    fold_cols = [f'fold_{i+1}' for i in range(NFOLDS) if f'fold_{i+1}' in feature_importances]
    if fold_cols:
        feature_importances['average'] = feature_importances[fold_cols].mean(axis=1)
    else:
        feature_importances['average'] = 0
    feature_importances.sort_values('average', ascending=False, inplace=True)

    return test_preds_scores, feature_importances

def main():
    train_df, test_df, sample_submission_df, test_ids_df = load_data()
    train_df_processed = preprocess_dataframe(train_df, is_train=True)
    test_df_processed = preprocess_dataframe(test_df, is_train=False)
    del train_df, test_df
    gc.collect()
    X, y, X_test, ranker_ids = prepare_matrices(train_df_processed, test_df_processed)
    del train_df_processed, test_df_processed
    gc.collect()

    newly_created_cols = ["dep_freq", "dep_is_hub"]
    cols_to_check = [c for c in newly_created_cols if c in X.columns]
    if cols_to_check:
        has_nans = X[cols_to_check].isna().any().any() or X_test[cols_to_check].isna().any().any()
        if has_nans:
            X = unify_nan_strategy(X)
            X_test = unify_nan_strategy(X_test)

    X, X_test, cat_features = encode_categoricals(X, X_test)
    # Use the same low_var_thresh across all scripts
    X, X_test, _ = clean_features(X, X_test, low_var_thresh=1)
    preds, _ = train_model(X, y, X_test, ranker_ids, cat_features)
    submission_df = test_ids_df.copy()
    submission_df['score'] = preds
    submission_df['selected'] = submission_df.groupby('ranker_id')['score'].rank(method='first', ascending=False).astype(int)
    submission_df = submission_df[['Id', 'ranker_id', 'selected']]
    submission_df.to_parquet('submission.parquet', index=False)
    submission_df.to_csv('submission.csv', index=False)
    print("Submission saved", submission_df.shape)
    return submission_df

if __name__ == "__main__":
    main()
