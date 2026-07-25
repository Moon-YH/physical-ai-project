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

    features = [
        "sleep_hours",
        "study_hours",
        "phone_hours",
        "exercise_hours",
        "productive_hours",
        "sleep_sufficient"
    ]

    input_data = df[features]
    target = df["score"]

    train_x, test_x, train_y, test_y = train_test_split(
        input_data,
        target,
        test_size=0.2,
        random_state=42
    )

    scale = StandardScaler()

    train_x = scale.fit_transform(train_x)
    test_x = scale.transform(test_x)

    print(f"학습 데이터: {train_x.shape}, 테스트 데이터: {test_x.shape}")

    return train_x, test_x, train_y, test_y, features


def train_and_evaluate(train_x, test_x, train_y, test_y):

    lr_model = LinearRegression()

    lr_model.fit(train_x, train_y)

    predict_score = lr_model.predict(test_x)

    mae = mean_absolute_error(test_y, predict_score)
    rmse = np.sqrt(mean_squared_error(test_y, predict_score))
    r2 = r2_score(test_y, predict_score)

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))

    return lr_model, predict_score


def plot_prediction(test_y, predict_score):

    plt.figure(figsize=(6, 6))

    plt.scatter(test_y, predict_score)

    min_score = test_y.min()

    if predict_score.min() < min_score:
        min_score = predict_score.min()

    max_score = test_y.max()

    if predict_score.max() > max_score:
        max_score = predict_score.max()

    plt.plot(
        [min_score, max_score],
        [min_score, max_score],
        "r--"
    )

    plt.xlabel("실제 점수")
    plt.ylabel("예측 점수")
    plt.title("실제 점수 vs 예측 점수")

    save_path = "outputs/model_evaluation.png"

    plt.savefig(save_path)
    plt.close()

    print(f"저장 완료: {save_path}")


def show_coefficients(lr_model, features):

    result_df = pd.DataFrame({
        "feature": features,
        "coefficient": lr_model.coef_
    })

    result_df = result_df.sort_values(by="coefficient")
    result_df = result_df[::-1]

    print("\n[피처별 회귀 계수]")
    print(result_df)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_clean_data(DATA_PATH)

    if df is None:
        return

    train_x, test_x, train_y, test_y, features = split_and_scale(df)

    lr_model, predict_score = train_and_evaluate(
        train_x,
        test_x,
        train_y,
        test_y
    )

    plot_prediction(test_y, predict_score)

    show_coefficients(lr_model, features)


if __name__ == "__main__":
    main()