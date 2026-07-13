import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_PATH = "data/student_habits_clean.csv"
OUTPUT_DIR = "outputs"


def load_clean_data(path):
    if not os.path.exists(path):
        print("정제 데이터 파일이 없습니다. 과제 2를 먼저 실행하세요.")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")

    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")

    return df


def plot_distributions(df):
    columns = [
        "sleep_hours",
        "study_hours",
        "phone_hours",
        "exercise_hours"
    ]

    titles = [
        "수면 시간 분포",
        "공부 시간 분포",
        "스마트폰 사용 시간 분포",
        "운동 시간 분포"
    ]

    x_labels = [
        "수면 시간",
        "공부 시간",
        "스마트폰 사용 시간",
        "운동 시간"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, col, title, x_label in zip(axes, columns, titles, x_labels):
        ax.hist(df[col], bins=15, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("빈도")

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "distribution.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"저장 완료: {output_path}")


def show_correlation(df):
    columns = [
        "sleep_hours",
        "study_hours",
        "phone_hours",
        "exercise_hours",
        "score"
    ]

    correlation = df[columns].corr()

    print("\n[상관계수 표]")
    print(correlation.round(2))

    return correlation


def plot_scatter(df):
    x_columns = [
        "study_hours",
        "sleep_hours",
        "phone_hours"
    ]

    titles = [
        "공부 시간과 점수",
        "수면 시간과 점수",
        "스마트폰 사용 시간과 점수"
    ]

    x_labels = [
        "공부 시간",
        "수면 시간",
        "스마트폰 사용 시간"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col, title, x_label in zip(axes, x_columns, titles, x_labels):
        ax.scatter(df[col], df["score"], alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("점수")

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "scatter_plots.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"저장 완료: {output_path}")


def plot_group_comparison(df):
    sleep_mean = df.groupby("sleep_sufficient")["score"].mean()
    phone_mean = df.groupby("phone_overuse")["score"].mean()

    print(f"\n수면 그룹 평균: {sleep_mean.round(1).to_dict()}")
    print(f"폰 그룹 평균: {phone_mean.round(1).to_dict()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        ["수면 부족", "수면 충족"],
        sleep_mean.reindex([0, 1]).values
    )
    axes[0].set_title("수면 충족 여부별 평균 점수")
    axes[0].set_xlabel("수면 그룹")
    axes[0].set_ylabel("평균 점수")

    axes[1].bar(
        ["정상 사용", "폰 과의존"],
        phone_mean.reindex([0, 1]).values
    )
    axes[1].set_title("스마트폰 과의존 여부별 평균 점수")
    axes[1].set_xlabel("스마트폰 사용 그룹")
    axes[1].set_ylabel("평균 점수")

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "group_comparison.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"저장 완료: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_clean_data(DATA_PATH)

    if df is None:
        return

    plot_distributions(df)
    show_correlation(df)
    plot_scatter(df)
    plot_group_comparison(df)


if __name__ == "__main__":
    main()