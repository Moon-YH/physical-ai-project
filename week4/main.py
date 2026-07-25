import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_PATH = "data/student_habits_clean.csv"
OUTPUT_DIR = "outputs"


def load_clean_data(path):

    if not os.path.exists(path):
        print("과제 2를 먼저 실행하세요.")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")

    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")

    return df


def split_and_scale(df):

    feature_cols = [
        "sleep_hours",
        "study_hours",
        "phone_hours",
        "exercise_hours",
        "productive_hours",
        "sleep_sufficient"
    ]

    X = df[feature_cols]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test):

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))

    return model, y_pred


def plot_prediction(y_test, y_pred):

    plt.figure(figsize=(6, 6))

    plt.scatter(y_test, y_pred)

    min_value = y_test.min()

    if y_pred.min() < min_value:
        min_value = y_pred.min()

    max_value = y_test.max()

    if y_pred.max() > max_value:
        max_value = y_pred.max()

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "r--"
    )

    plt.xlabel("실제 점수")
    plt.ylabel("예측 점수")
    plt.title("실제 점수 vs 예측 점수")

    output_path = "outputs/model_evaluation.png"

    plt.savefig(output_path)
    plt.close()

    print(f"저장 완료: {output_path}")


def show_coefficients(model, feature_cols):

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_
    })

    coef_df = coef_df.sort_values(by="coefficient")
    coef_df = coef_df[::-1]

    print("\n[피처별 회귀 계수]")
    print(coef_df)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_clean_data(DATA_PATH)

    if df is None:
        return

    X_train, X_test, y_train, y_test, feature_cols = split_and_scale(df)

    model, y_pred = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test
    )

    plot_prediction(y_test, y_pred)

    show_coefficients(model, feature_cols)


if __name__ == "__main__":
    main()