import pandas as pd
import numpy as np

class InvalidEngineHistoryError(ValueError):
    pass

def history_to_dataframe(request):
    rows = []

    for observation in request.history:
        rows.append(observation.model_dump())

    dataframe = pd.DataFrame(rows)
    dataframe["unit_number"] = request.engine_id

    return dataframe

def prepare_feature_input(dataframe):
    dataframe = dataframe.copy()

    # Rename sensor columns here.
    for column in dataframe.columns:
        if column.startswith("sensor_measurement_"):
            dataframe = dataframe.rename(
                columns={column: "current_" + column}
            )

    return dataframe.sort_values(
        ["unit_number", "time_in_cycles"]
    ).reset_index(drop=True)    

def add_time_series_features(dataframe, sensors):
    dataframe = dataframe.copy()

    rolling_means = (
        dataframe.groupby(["unit_number"])[sensors]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_means = rolling_means.rename(
        columns={
            column: column[8:] + "_mean_20"
            for column in rolling_means.columns
        }
    )
    dataframe = pd.concat([dataframe, rolling_means], axis=1)

    rolling_stds = (
        dataframe.groupby("unit_number")[sensors]
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    rolling_stds = rolling_stds.rename(
        columns={
            column: column[8:] + "_std_20"
            for column in rolling_stds.columns
        }
    )
    dataframe = pd.concat([dataframe, rolling_stds], axis=1)

    rolling_min = (
        dataframe.groupby("unit_number")[sensors]
        .rolling(window=20, min_periods=20)
        .min()
        .reset_index(level=0, drop=True)
    )
    rolling_min = rolling_min.rename(
        columns={
            column: column[8:] + "_min_20"
            for column in rolling_min.columns
        }
    )
    dataframe = pd.concat([dataframe, rolling_min], axis=1)

    rolling_max = (
        dataframe.groupby("unit_number")[sensors]
        .rolling(window=20, min_periods=20)
        .max()
        .reset_index(level=0, drop=True)
    )
    rolling_max = rolling_max.rename(
        columns={
            column: column[8:] + "_max_20"
            for column in rolling_max.columns
        }
    )
    dataframe = pd.concat([dataframe, rolling_max], axis=1)

    difference = dataframe.groupby("unit_number")[sensors].diff()
    difference = difference.rename(
        columns={
            column: column[8:] + "_difference"
            for column in difference.columns
        }
    )
    dataframe = pd.concat([dataframe, difference], axis=1)

    percentage_change = (
        dataframe.groupby("unit_number")[sensors]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
    )
    percentage_change = percentage_change.rename(
        columns={
            column: column[8:] + "_%change"
            for column in percentage_change.columns
        }
    )
    dataframe = pd.concat([dataframe, percentage_change], axis=1)

    sensor_slope = (
        dataframe.groupby("unit_number")[sensors]
        .rolling(window=20, min_periods=20)
        .apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
            raw=True,
        )
        .reset_index(level=0, drop=True)
    )
    sensor_slope = sensor_slope.rename(
        columns={
            column: column[8:] + "_sensor_slope_20"
            for column in sensor_slope.columns
        }
    )
    dataframe = pd.concat([dataframe, sensor_slope], axis=1)

    ema_features = (
        dataframe.groupby("unit_number")[sensors]
        .transform(
            lambda column: column.ewm(
                span=20,
                adjust=False,
                min_periods=1,
            ).mean()
        )
    )
    ema_features = ema_features.rename(
        columns={
            column: column[8:] + "_ema_20"
            for column in ema_features.columns
        }
    )
    dataframe = pd.concat([dataframe, ema_features], axis=1)

    return dataframe


def prepare_model_input(request, feature_columns):
    dataframe = history_to_dataframe(request)
    dataframe = prepare_feature_input(dataframe)

    sensors = [
        column
        for column in dataframe.columns
        if column.startswith("current_sensor_measurement_")
    ]

    dataframe = add_time_series_features(dataframe, sensors)
    latest_row = dataframe.tail(1)

    missing_features = [
        column
        for column in feature_columns
        if column not in latest_row.columns
    ]

    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    # Select training features in order, excluding removed constants and IDs.
    model_input = latest_row.loc[:, feature_columns].copy()

    if model_input.isna().any().any():
        raise InvalidEngineHistoryError(
            "Engine history produces missing model features, "
            "possibly from undefined percentage changes."
        )

    if not np.isfinite(model_input.to_numpy(dtype=float)).all():
        raise InvalidEngineHistoryError(
            "Engine history produces non-finite model features."
        )

    return model_input