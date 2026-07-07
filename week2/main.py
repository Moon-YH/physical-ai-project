import pandas as pd

DATA_PATH = "data/student_habits.csv"
OUTPUT_PATH = "data/student_habits_clean.csv"


def load_data(path):
    df = pd.read_csv(path)
    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
    return df


def handle_missing(df):
    missing_cols = ["sleep_hours", "phone_hours", "exercise_hours"]

    for col in missing_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    total_missing = df.isnull().sum().sum()
    print(f"결측치 처리 후 남은 결측치: {total_missing} 개")

    return df


def handle_outliers(df):
    numeric_cols = ["sleep_hours", "study_hours", "phone_hours", "exercise_hours"]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower_bound, upper_bound)

    print("이상치 처리(클리핑) 완료")

    return df


def convert_types(df):
    df["gender"] = df["gender"].str.strip()
    df["gender_code"] = (df["gender"] == "여").astype(int)

    print("gender → gender_code 인코딩 완료")

    return df


def add_features(df):
    df["productive_hours"] = df["study_hours"] + df["exercise_hours"]
    df["sleep_sufficient"] = (df["sleep_hours"] >= 7).astype(int)
    df["phone_overuse"] = (df["phone_hours"] > 4).astype(int)

    print("파생변수 생성 완료: productive_hours, sleep_sufficient, phone_overuse")

    return df


def main():
    df = load_data(DATA_PATH)

    df = handle_missing(df)
    df = handle_outliers(df)
    df = convert_types(df)
    df = add_features(df)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"정제 데이터 저장 완료: {OUTPUT_PATH} ({df.shape[0]}행 × {df.shape[1]}열)")


if __name__ == "__main__":
    main()