import pandas as pd
import numpy as np
import wfdb
import ast
import tsfel
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_raw_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def get_data_ready_for_training(
    dataset_path: str,
    sampling_rate: int = 100,
    use_tsfel: bool = False,
    use_temporal_features: bool = False,
    scale_features: bool = False,
    input_3D: bool = False,
    train_val_split: bool = True,
) -> (
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MultiLabelBinarizer,
):

    # data loading
    y_data = pd.read_csv(dataset_path + "ptbxl_database.csv", index_col="ecg_id")
    y_data.scp_codes = y_data.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(dataset_path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    y_data["diagnostic_superclass"] = y_data.scp_codes.apply(aggregate_diagnostic)

    print("Loaded labels")

    # data filtering
    y = y_data[y_data["diagnostic_superclass"].apply(lambda x: len(x) >= 1)]
    X = load_raw_data(y, sampling_rate, dataset_path)

    print("Loaded ECG signals")

    # data split into train and test
    test_fold = 10
    X_train = X[np.where(y.strat_fold != test_fold)]
    y_train = y[(y.strat_fold != test_fold)].diagnostic_superclass
    X_test = X[np.where(y.strat_fold == test_fold)]
    y_test = y[y.strat_fold == test_fold].diagnostic_superclass

    # data label encoding
    multi_label_binarizer = MultiLabelBinarizer()
    multi_label_binarizer.fit(y_train)

    y_train = multi_label_binarizer.transform(y_train)
    y_test = multi_label_binarizer.transform(y_test)

    print("Encoded labels")

    # prepare data for classic machine learning
    if use_tsfel:
        domain = "temporal" if use_temporal_features else None
        cfg_file = tsfel.get_features_by_domain(domain)
        X_train = tsfel.time_series_features_extractor(
            dict_features=cfg_file, signal_windows=X_train, fs=sampling_rate
        )
        X_test = tsfel.time_series_features_extractor(
            dict_features=cfg_file, signal_windows=X_test, fs=sampling_rate
        )

        # Highly correlated features are removed
        corr_features = tsfel.correlated_features(X_train)
        X_train.drop(corr_features, axis=1, inplace=True)
        X_test.drop(corr_features, axis=1, inplace=True)

        # Remove low variance features
        selector = VarianceThreshold()
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test)

        print("Extracted features")

    # scaling features
    if scale_features:
        if not use_tsfel:
            num_train_patients, num_time_steps, num_features = X_train.shape
            num_test_patients = X_test.shape[0]

            X_train = np.reshape(X_train, newshape=(-1, num_features))
            X_test = np.reshape(X_test, newshape=(-1, num_features))

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        print("Scaled ECG signals")

        if not use_tsfel:
            X_train = np.reshape(
                X_train, newshape=(num_train_patients, num_time_steps, num_features)
            )
            X_test = np.reshape(
                X_test, newshape=(num_test_patients, num_time_steps, num_features)
            )

    if all([input_3D, not use_tsfel]):
        X_train = np.reshape(
            X_train, newshape=(num_train_patients, num_time_steps, num_features, 1)
        )
        X_test = np.reshape(
            X_test, newshape=(num_test_patients, num_time_steps, num_features, 1)
        )
        print(f"Reshaped ECG signals to: {X_train.shape[1:]}")

    X_val = None
    y_val = None

    if train_val_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )
        print("Split training data into training|validation")

    return X_train, X_val, X_test, y_train, y_val, y_test, multi_label_binarizer
