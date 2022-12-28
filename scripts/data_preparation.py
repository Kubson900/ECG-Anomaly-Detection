import pandas as pd
import numpy as np
import wfdb
import ast

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
    sampling_rate: int,
    dataset_path: str,
    input_3D: bool = True,
    scale_features: bool = True,
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

    num_train_patients, num_time_steps, num_features = X_train.shape
    num_test_patients = X_test.shape[0]

    # scaling features
    if scale_features:
        scaler = StandardScaler()

        X_train = np.reshape(X_train, newshape=(-1, num_features))
        X_test = np.reshape(X_test, newshape=(-1, num_features))

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        print("Scaled ECG signals")

    if input_3D:
        X_train = np.reshape(
            X_train, newshape=(num_train_patients, num_time_steps, num_features, 1)
        )
        X_test = np.reshape(
            X_test, newshape=(num_test_patients, num_time_steps, num_features, 1)
        )
    else:
        X_train = np.reshape(
            X_train, newshape=(num_train_patients, num_time_steps, num_features)
        )
        X_test = np.reshape(
            X_test, newshape=(num_test_patients, num_time_steps, num_features)
        )
    print(f"Reshaped ECG signals to: {X_train.shape[1:]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    print("Split training data into training|validation")

    return X_train, X_val, X_test, y_train, y_val, y_test, multi_label_binarizer
