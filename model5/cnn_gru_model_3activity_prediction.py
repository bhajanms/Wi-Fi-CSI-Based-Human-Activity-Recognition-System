# =========================================================
# REALTIME WIFI CSI HAR LIVE PREDICTION
# ACTIVITIES:
# WALK / STATIC / FALL
#
# FINAL FULLY CORRECTED VERSION
# =========================================================

import serial
import re
import time
import pickle
import requests

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt

from tensorflow.keras.models import load_model

# =========================================================
# CONFIG
# =========================================================

SERIAL_PORT_RT = "COM10"

SERIAL_BAUDRATE_RT = 115200

MODEL_FOLDER_RT = (
    r"C:\esp32-csi-tool\model_fall_rt"
)

# =========================================================
# REALTIME SETTINGS
# =========================================================

LIVE_PREDICTION_INTERVAL_RT = 1

CSI_WINDOW_LENGTH_RT = 100

CSI_MOVING_AVERAGE_RT = 5

RAW_PACKET_WINDOW_RT = (
    CSI_WINDOW_LENGTH_RT
    + CSI_MOVING_AVERAGE_RT
    - 1
)

# =========================================================
# FALL SENSITIVITY
# =========================================================

FALL_CONFIDENCE_THRESHOLD_RT = 0.30

# =========================================================
# API SETTINGS
# =========================================================

ENABLE_API_RT = True

# SAME PC API
API_ENDPOINT_RT = (
    "http://127.0.0.1:5000/activity"
)

# =========================================================
# FLOAT SAFETY
# =========================================================

np.seterr(all='ignore')

# =========================================================
# LOAD MODEL + PREPROCESSORS
# =========================================================

print("\nLoading Model...\n")

realtime_live_model = load_model(
    MODEL_FOLDER_RT
    + r"\realtime_cnn_gru_model.keras"
)

realtime_scaler = pickle.load(
    open(
        MODEL_FOLDER_RT
        + r"\rt_scaler.pkl",
        "rb"
    )
)

realtime_pca = pickle.load(
    open(
        MODEL_FOLDER_RT
        + r"\rt_pca.pkl",
        "rb"
    )
)

expected_csi_length_rt = pickle.load(
    open(
        MODEL_FOLDER_RT
        + r"\rt_csi_length.pkl",
        "rb"
    )
)

realtime_labels = pickle.load(
    open(
        MODEL_FOLDER_RT
        + r"\rt_labels.pkl",
        "rb"
    )
)

print("Model Loaded Successfully")

print(
    "Expected CSI Length :",
    expected_csi_length_rt
)

print(
    "Labels :",
    realtime_labels
)

# =========================================================
# SERIAL CONNECTION
# =========================================================

print("\nOpening Serial Port...\n")

serial_connection_rt = serial.Serial(
    SERIAL_PORT_RT,
    SERIAL_BAUDRATE_RT,
    timeout=1
)

print(
    "Listening on",
    SERIAL_PORT_RT
)

# =========================================================
# BUTTER FILTER
# =========================================================

def apply_realtime_filter(
    data,
    cutoff=0.1,
    order=3
):

    b, a = butter(order, cutoff)

    return filtfilt(
        b,
        a,
        data,
        axis=0
    )

# =========================================================
# PARSE CSI
# =========================================================

def parse_realtime_csi(line):

    try:

        matched = re.search(
            r"\[(.*)\]",
            line
        )

        if not matched:
            return None

        row = matched.group(1)

        values = np.array(
            [
                float(x)
                for x in re.findall(
                    r"-?\d+",
                    row
                )
            ],
            dtype=np.float32
        )

        return values

    except:

        return None

# =========================================================
# AMP + PHASE
# =========================================================

def convert_realtime_amp_phase(csi):

    # ESP32 CSI FORMAT
    # imag, real, imag, real

    imag = csi[::2]

    real = csi[1::2]

    amplitude = np.sqrt(
        imag**2 + real**2
    )

    phase = np.unwrap(
        np.arctan2(imag, real)
    )

    return amplitude, phase

# =========================================================
# SUBCARRIER SELECTION
# =========================================================

def select_realtime_subcarriers(df):

    left = df.iloc[:, 5:32]

    right = df.iloc[:, 33:60]

    return pd.concat(
        [left, right],
        axis=1
    )

# =========================================================
# MOVING AVERAGE
# =========================================================

def smooth_realtime_signal(df):

    return df.rolling(
        CSI_MOVING_AVERAGE_RT
    ).mean().dropna()

# =========================================================
# BUFFERS
# =========================================================

amplitude_buffer_rt = []

phase_buffer_rt = []

last_prediction_time_rt = time.time()

# =========================================================
# LIVE LOOP
# =========================================================

print("\nStarting Live Prediction...\n")

while True:

    try:

        # =================================================
        # READ SERIAL
        # =================================================

        serial_line_rt = (
            serial_connection_rt
            .readline()
            .decode(errors="ignore")
        )

        # ONLY CSI DATA
        if "CSI_DATA" not in serial_line_rt:
            continue

        parsed_packet_rt = parse_realtime_csi(
            serial_line_rt
        )

        if parsed_packet_rt is None:
            continue

        # =================================================
        # FIXED CSI LENGTH
        # =================================================

        if (
            len(parsed_packet_rt)
            != expected_csi_length_rt
        ):

            continue

        # =================================================
        # AMP + PHASE
        # =================================================

        amplitude_rt, phase_rt = (
            convert_realtime_amp_phase(
                parsed_packet_rt
            )
        )

        amplitude_buffer_rt.append(
            amplitude_rt
        )

        phase_buffer_rt.append(
            phase_rt
        )

        # =================================================
        # FIXED BUFFER
        # =================================================

        if (
            len(amplitude_buffer_rt)
            > RAW_PACKET_WINDOW_RT
        ):

            amplitude_buffer_rt.pop(0)

            phase_buffer_rt.pop(0)

        # =================================================
        # WAIT FOR BUFFER
        # =================================================

        if (
            len(amplitude_buffer_rt)
            < RAW_PACKET_WINDOW_RT
        ):

            continue

        # =================================================
        # PREDICTION INTERVAL
        # =================================================

        if (
            time.time()
            - last_prediction_time_rt
            >= LIVE_PREDICTION_INTERVAL_RT
        ):

            # =============================================
            # DATAFRAME
            # =============================================

            amplitude_df_rt = pd.DataFrame(
                amplitude_buffer_rt
            )

            phase_df_rt = pd.DataFrame(
                phase_buffer_rt
            )

            # =============================================
            # SUBCARRIER SELECTION
            # =============================================

            amplitude_df_rt = (
                select_realtime_subcarriers(
                    amplitude_df_rt
                )
            )

            phase_df_rt = (
                select_realtime_subcarriers(
                    phase_df_rt
                )
            )

            # =============================================
            # MOVING AVERAGE
            # =============================================

            amplitude_df_rt = (
                smooth_realtime_signal(
                    amplitude_df_rt
                )
            )

            phase_df_rt = (
                smooth_realtime_signal(
                    phase_df_rt
                )
            )

            # =============================================
            # EXACT WINDOW CHECK
            # =============================================

            if (
                len(amplitude_df_rt)
                != CSI_WINDOW_LENGTH_RT
            ):

                continue

            # =============================================
            # STATIC REMOVAL
            # =============================================

            amplitude_df_rt = (
                amplitude_df_rt
                - amplitude_df_rt.mean(axis=0)
            )

            phase_df_rt = (
                phase_df_rt
                - phase_df_rt.mean(axis=0)
            )

            # =============================================
            # FEATURE CONCAT
            # =============================================

            combined_features_rt = pd.concat(
                [
                    amplitude_df_rt,
                    phase_df_rt
                ],
                axis=1
            )

            # =============================================
            # BUTTER FILTER
            # =============================================

            filtered_features_rt = (
                apply_realtime_filter(
                    combined_features_rt.values
                )
            )

            # =============================================
            # SAFE VALUES
            # =============================================

            filtered_features_rt = np.nan_to_num(
                filtered_features_rt
            )

            # =============================================
            # SCALE
            # =============================================

            scaled_features_rt = (
                realtime_scaler.transform(
                    filtered_features_rt
                )
            )

            # =============================================
            # PCA
            # =============================================

            reduced_features_rt = (
                realtime_pca.transform(
                    scaled_features_rt
                )
            )

            # =============================================
            # RESHAPE
            # =============================================

            X_live_rt = np.expand_dims(
                reduced_features_rt,
                axis=0
            ).astype(np.float32)

            # =============================================
            # SAFE CHECK
            # =============================================

            if (
                np.isnan(X_live_rt).any()
                or
                np.isinf(X_live_rt).any()
            ):

                continue

            # =============================================
            # MODEL PREDICTION
            # =============================================

            prediction_scores_rt = (
                realtime_live_model.predict(
                    X_live_rt,
                    verbose=0
                )[0]
            )

            predicted_index_rt = int(
                np.argmax(
                    prediction_scores_rt
                )
            )

            predicted_activity_rt = (
                realtime_labels[
                    predicted_index_rt
                ]
            )

            prediction_confidence_rt = float(
                prediction_scores_rt[
                    predicted_index_rt
                ]
            )

            # =============================================
            # INDIVIDUAL PROBABILITIES
            # =============================================

            walk_probability_rt = float(
                prediction_scores_rt[0]
            )

            static_probability_rt = float(
                prediction_scores_rt[1]
            )

            fall_probability_rt = float(
                prediction_scores_rt[2]
            )

            # =============================================
            # FALL PRIORITY LOGIC
            # =============================================

            if (
                fall_probability_rt
                >= FALL_CONFIDENCE_THRESHOLD_RT
            ):

                predicted_activity_rt = "fall"

                prediction_confidence_rt = (
                    fall_probability_rt
                )

            # =============================================
            # PRINT RESULTS
            # =============================================

            print("\n================================")

            print(
                "Predicted Activity :",
                predicted_activity_rt
            )

            print(
                "Confidence         :",
                round(
                    prediction_confidence_rt,
                    3
                )
            )

            print("\nProbabilities")

            print(
                "walk   :",
                round(
                    walk_probability_rt,
                    3
                )
            )

            print(
                "static :",
                round(
                    static_probability_rt,
                    3
                )
            )

            print(
                "fall   :",
                round(
                    fall_probability_rt,
                    3
                )
            )

            print("================================")

            # =============================================
            # API SEND
            # =============================================

            if ENABLE_API_RT:

                try:

                    payload = {

                        "activity":
                        predicted_activity_rt,

                        "confidence":
                        prediction_confidence_rt,

                        "walk_probability":
                        walk_probability_rt,

                        "static_probability":
                        static_probability_rt,

                        "fall_probability":
                        fall_probability_rt
                    }

                    response = requests.post(
                        API_ENDPOINT_RT,
                        json=payload,
                        timeout=5
                    )

                    print(
                        "API Status :",
                        response.status_code
                    )

                except Exception as api_error_rt:

                    print(
                        "API Error :",
                        api_error_rt
                    )

            # =============================================
            # UPDATE TIMER
            # =============================================

            last_prediction_time_rt = (
                time.time()
            )

    except KeyboardInterrupt:

        print(
            "\nStopped Live Prediction"
        )

        break

    except Exception as runtime_error_rt:

        print(
            "\nRuntime Error :",
            runtime_error_rt
        )