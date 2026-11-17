import io
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="당뇨병 발병 여부 예측",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 건강검진 데이터로 당뇨병 발병 여부 예측하기")

st.write(
    "건강검진 데이터를 탐색하고 "
    "당뇨병 발병 여부를 예측하는 실습 앱입니다."
)

DATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "ehababoelnaga/diabetes-dataset"
)

ALIASES = {
    "나이": ["age"],
    "성별": ["gender", "sex"],
    "혈당": ["glucose"],
    "BMI": ["bmi"],
    "혈압": ["bloodpressure", "blood pressure", "blood_pressure"],
    "당뇨병 발병 여부": [
        "diabetes", "outcome", "class", "target"
    ]
}


# --------------------------------------------------
# 데이터 관련 함수
# --------------------------------------------------

def normalize_name(name):
    return (
        str(name).strip().lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_column(columns, candidates):
    normalized = {
        normalize_name(c): c for c in columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    compact = {
        k.replace(" ", ""): v
        for k, v in normalized.items()
    }

    for candidate in candidates:
        key = normalize_name(candidate).replace(" ", "")

        if key in compact:
            return compact[key]

    return None


def read_csv_bytes(raw_bytes):
    for encoding in ["utf-8", "utf-8-sig", "cp949", "latin1"]:
        try:
            return pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=encoding
            )
        except Exception:
            pass

    raise ValueError("CSV 파일을 읽을 수 없습니다.")


@st.cache_data(ttl=3600)
def download_data():
    response = requests.get(
        DATA_URL,
        timeout=40
    )

    response.raise_for_status()

    content = response.content

    if zipfile.is_zipfile(io.BytesIO(content)):

        with zipfile.ZipFile(io.BytesIO(content)) as archive:

            files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
                and not name.startswith("__MACOSX")
            ]

            if not files:
                raise ValueError("CSV 파일을 찾지 못했습니다.")

            preferred = [
                name for name in files
                if any(
                    word in name.lower()
                    for word in ["training", "train", "diabetes"]
                )
            ]

            filename = preferred[0] if preferred else files[0]

            with archive.open(filename) as file:
                df = read_csv_bytes(file.read())

            return df, filename

    return read_csv_bytes(content), "Kaggle CSV"


def convert_target(series):
    """
    당뇨병 여부를 0과 1로 변환합니다.
    0 = 정상
    1 = 당뇨병
    """

    values = series.dropna().astype(str).str.strip()
    unique = list(pd.unique(values))

    if len(unique) != 2:
        raise ValueError(
            "당뇨병 여부 컬럼에는 두 종류의 값이 있어야 합니다."
        )

    positive_words = [
        "1", "yes", "true", "positive",
        "diabetes", "diabetic"
    ]

    negative_words = [
        "0", "no", "false", "negative",
        "normal", "non-diabetic"
    ]

    positive = None
    negative = None

    for value in unique:

        if value.lower() in positive_words:
            positive = value

        if value.lower() in negative_words:
            negative = value

    if positive is not None and negative is not None:

        mapping = {
            negative: 0,
            positive: 1
        }

    else:

        try:
            numbers = sorted(float(v) for v in unique)

            if numbers == [0.0, 1.0]:
                mapping = {
                    v: int(float(v))
                    for v in unique
                }
            else:
                ordered = sorted(unique)
                mapping = {
                    ordered[0]: 0,
                    ordered[1]: 1
                }

        except Exception:
            ordered = sorted(unique)

            mapping = {
                ordered[0]: 0,
                ordered[1]: 1
            }

    converted = (
        series.astype(str)
        .str.strip()
        .map(mapping)
    )

    return converted, mapping


def make_groups(series, bins):
    """
    숫자형 변수는 분위수 기준으로 구간을 나눕니다.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    if numeric.notna().sum() == 0:
        return series.fillna("정보 없음").astype(str)

    if numeric.nunique() <= 1:
        return numeric.astype(str)

    try:
        grouped = pd.qcut(
            numeric,
            q=bins,
            duplicates="drop"
        )

    except Exception:
        grouped = pd.cut(
            numeric,
            bins=bins,
            duplicates="drop"
        )

    return grouped.astype(str).replace(
        "nan",
        "정보 없음"
    ).fillna("정보 없음")


def diabetes_rate_table(data, group_column):
    """
    그룹별 전체 인원, 당뇨병 환자 수, 발병 비율 계산
    """

    result = data.groupby(
        group_column,
        observed=True
    )["_target"].agg(
        전체_인원="count",
        당뇨병_환자수="sum"
    ).reset_index()

    result["당뇨병_발병_비율(%)"] = (
        result["당뇨병_환자수"]
        / result["전체_인원"]
        * 100
    )

    result["당뇨병_환자수"] = (
        result["당뇨병_환자수"].astype(int)
    )

    result["당뇨병_발병_비율(%)"] = (
        result["당뇨병_발병_비율(%)"].round(1)
    )

    return result


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

st.sidebar.header("📂 데이터 준비")

uploaded = st.sidebar.file_uploader(
    "CSV 파일 직접 업로드",
    type=["csv"]
)

mode = st.sidebar.radio(
    "데이터 불러오기",
    [
        "Kaggle에서 자동 불러오기",
        "CSV 파일 직접 업로드"
    ]
)

df = None

if mode == "Kaggle에서 자동 불러오기":

    try:
        with st.spinner("데이터를 불러오는 중입니다..."):
            df, source = download_data()

    except Exception as e:

        st.sidebar.error(
            "Kaggle 자동 다운로드에 실패했습니다."
        )

        st.sidebar.caption(str(e))

        st.sidebar.markdown(
            "[Kaggle 데이터 다운로드](https://www.kaggle.com/datasets/ehababoelnaga/diabetes-dataset)"
        )

        if uploaded is not None:
            df = read_csv_bytes(uploaded.getvalue())

else:

    if uploaded is not None:
        df = read_csv_bytes(uploaded.getvalue())


if df is None:
    st.info(
        "왼쪽 메뉴에서 CSV 파일을 업로드하거나 "
        "Kaggle 자동 불러오기를 이용해 주세요."
    )
    st.stop()


df = df.copy()

df.columns = [
    str(c).strip()
    for c in df.columns
]

df = df.dropna(how="all")

st.sidebar.success("데이터 불러오기 완료")

st.sidebar.write(
    f"전체 데이터: {len(df):,}명"
)


# --------------------------------------------------
# 컬럼 확인
# --------------------------------------------------

detected = {
    label: find_column(df.columns, names)
    for label, names in ALIASES.items()
}

st.sidebar.divider()
st.sidebar.subheader("분석 설정")

target_options = list(df.columns)

default_target = detected["당뇨병 발병 여부"]

target_index = (
    target_options.index(default_target)
    if default_target in target_options
    else 0
)

target_col = st.sidebar.selectbox(
    "당뇨병 발병 여부 컬럼",
    target_options,
    index=target_index
)


# --------------------------------------------------
# 목표값 변환
# --------------------------------------------------

try:
    target, target_mapping = convert_target(
        df[target_col]
    )

except Exception as e:
    st.error(f"당뇨병 발병 여부 컬럼 오류: {e}")
    st.stop()


data = df.copy()

data["_target"] = target

data = data.dropna(
    subset=["_target"]
)

data["_target"] = data["_target"].astype(int)


# --------------------------------------------------
# 탭 구성
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 데이터 살펴보기",
    "📊 데이터 탐색",
    "🤖 분류 모델",
    "🔎 직접 예측하기"
])


# ==================================================
# 1. 데이터 살펴보기
# ==================================================

with tab1:

    st.header("1. 데이터 살펴보기")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "전체 인원",
        f"{len(df):,}명"
    )

    c2.metric(
        "속성 개수",
        f"{len(df.columns):,}개"
    )

    c3.metric(
        "빈 값 개수",
        f"{int(df.isna().sum().sum()):,}개"
    )

    st.subheader("데이터 미리보기")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("컬럼별 정보")

    meanings = {
        "나이": "건강검진을 받은 사람의 나이",
        "성별": "건강검진을 받은 사람의 성별",
        "혈당": "혈액 속 포도당의 농도",
        "BMI": "키와 몸무게를 이용해 계산한 비만 정도",
        "혈압": "혈액이 혈관 벽에 가하는 압력",
        "당뇨병 발병 여부": "당뇨병 발병 여부(1: 당뇨병, 0: 정상)"
    }

    rows = []

    for col in df.columns:

        role = next(
            (
                label
                for label, found in detected.items()
                if found == col
            ),
            ""
        )

        rows.append({
            "컬럼명": col,
            "우리말 뜻": meanings.get(role, ""),
            "자료형": str(df[col].dtype),
            "빈 값 개수": int(df[col].isna().sum()),
            "고유값 개수": int(df[col].nunique())
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("기초 통계")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )


# ==================================================
# 2. 데이터 탐색
# ==================================================

with tab2:

    st.header("2. 데이터 탐색하기")

    st.write(
        "독립 변수를 선택하면 해당 변수의 값에 따라 "
        "당뇨병 발병 비율이 어떻게 달라지는지 확인할 수 있습니다."
    )

    # 전체 당뇨병 비율

    total_people = len(data)

    diabetic_people = int(
        data["_target"].sum()
    )

    total_rate = (
        diabetic_people / total_people * 100
        if total_people > 0
        else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "전체 인원",
        f"{total_people:,}명"
    )

    c2.metric(
        "당뇨병 환자",
        f"{diabetic_people:,}명"
    )

    c3.metric(
        "전체 당뇨병 발병 비율",
        f"{total_rate:.1f}%"
    )

    st.divider()

    # 독립 변수 선택

    st.subheader("① 분석할 독립 변수 선택")

    feature_options = [
        c for c in df.columns
        if c != target_col
    ]

    default_features = []

    for key in ["나이", "성별", "혈당", "BMI", "혈압"]:

        col = detected.get(key)

        if col in feature_options:
            default_features.append(col)

    selected_features = st.multiselect(
        "비교하고 싶은 독립 변수를 선택하세요.",
        feature_options,
        default=default_features
    )

    bins = st.slider(
        "숫자형 변수의 구간 개수",
        min_value=2,
        max_value=10,
        value=5
    )

    if not selected_features:

        st.info(
            "독립 변수를 한 개 이상 선택하면 "
            "당뇨병 발병 비율을 분석할 수 있습니다."
        )

    else:

        st.divider()

        st.subheader("② 독립 변수별 당뇨병 발병 비율")

        st.caption(
            "각 그룹에서 당뇨병 환자 수를 전체 인원으로 나누어 계산합니다."
        )

        for feature in selected_features:

            st.markdown(f"### 📌 {feature}에 따른 당뇨병 발병 비율")

            temp = data[[feature, "_target"]].copy()

            # 숫자형 변수인지 확인

            numeric_values = pd.to_numeric(
                temp[feature],
                errors="coerce"
            )

            is_numeric = (
                numeric_values.notna().sum() > 0
                and numeric_values.notna().sum()
                >= temp[feature].notna().sum() * 0.8
            )

            group_col = "_분석그룹"

            if is_numeric:

                temp[group_col] = make_groups(
                    numeric_values,
                    bins
                )

            else:

                temp[group_col] = (
                    temp[feature]
                    .fillna("정보 없음")
                    .astype(str)
                )

            result = diabetes_rate_table(
                temp,
                group_col
            )

            result = result.rename(
                columns={
                    group_col: feature
                }
            )

            # 표 출력

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            # 그래프 출력

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.bar(
                result[feature].astype(str),
                result["당뇨병_발병_비율(%)"]
            )

            ax.axhline(
                total_rate,
                linestyle="--",
                label=f"전체 평균 {total_rate:.1f}%"
            )

            ax.set_title(
                f"{feature}에 따른 당뇨병 발병 비율"
            )

            ax.set_xlabel(feature)
            ax.set_ylabel("당뇨병 발병 비율 (%)")

            ax.set_ylim(
                0,
                max(
                    100,
                    result["당뇨병_발병_비율(%)"].max() + 10
                )
            )

            ax.tick_params(
                axis="x",
                rotation=35
            )

            ax.legend()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.divider()

        # --------------------------------------------------
        # 여러 독립 변수의 조합 분석
        # --------------------------------------------------

        st.subheader("③ 여러 독립 변수를 함께 분석하기")

        st.write(
            "선택한 독립 변수들을 조합했을 때 "
            "각 조건에 해당하는 사람들의 당뇨병 발병 비율을 확인합니다."
        )

        use_combination = st.checkbox(
            "선택한 독립 변수의 조합별 발병 비율도 확인하기"
        )

        if use_combination:

            combo_data = data[
                selected_features + ["_target"]
            ].copy()

            combo_columns = []

            for feature in selected_features:

                numeric_values = pd.to_numeric(
                    combo_data[feature],
                    errors="coerce"
                )

                is_numeric = (
                    numeric_values.notna().sum() > 0
                    and numeric_values.notna().sum()
                    >= combo_data[feature].notna().sum() * 0.8
                )

                new_col = f"{feature}_그룹"

                if is_numeric:

                    combo_data[new_col] = make_groups(
                        numeric_values,
                        bins
                    )

                else:

                    combo_data[new_col] = (
                        combo_data[feature]
                        .fillna("정보 없음")
                        .astype(str)
                    )

                combo_columns.append(new_col)

            combination_result = (
                combo_data
                .groupby(
                    combo_columns,
                    observed=True
                )["_target"]
                .agg(
                    전체_인원="count",
                    당뇨병_환자수="sum"
                )
                .reset_index()
            )

            combination_result["당뇨병_발병_비율(%)"] = (
                combination_result["당뇨병_환자수"]
                / combination_result["전체_인원"]
                * 100
            ).round(1)

            combination_result = combination_result.sort_values(
                "당뇨병_발병_비율(%)",
                ascending=False
            )

            # 표에 보기 좋은 컬럼명 적용

            combination_result = combination_result.rename(
                columns={
                    f"{feature}_그룹": feature
                    for feature in selected_features
                }
            )

            st.write("당뇨병 발병 비율이 높은 조합부터 표시합니다.")

            st.dataframe(
                combination_result.head(30),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "조합별 인원이 적으면 발병 비율이 크게 달라질 수 있습니다. "
                "전체 인원도 함께 확인하세요."
            )


# ==================================================
# 3. 분류 모델
# ==================================================

with tab3:

    st.header("3. 분류 모델 만들기")

    st.write(
        "선택한 독립 변수를 이용하여 "
        "당뇨병 발병 여부를 예측하는 모델을 만듭니다."
    )

    st.info(
        "당뇨병(1)을 양성으로 설정합니다."
    )

    feature_options = [
        c for c in df.columns
        if c != target_col
    ]

    default_features = [
        detected[key]
        for key in ["나이", "혈당", "BMI", "혈압"]
        if detected.get(key) in feature_options
    ]

    if not default_features:
        default_features = feature_options[:4]

    selected_model_features = st.multiselect(
        "모델의 독립 변수 선택",
        feature_options,
        default=default_features,
        key="model_features"
    )

    test_size = st.slider(
        "시험용 데이터 비율 (%)",
        10,
        40,
        20,
        5
    )

    random_seed = st.number_input(
        "데이터 분할 기준 숫자",
        min_value=0,
        max_value=9999,
        value=42
    )

    if st.button(
        "모델 학습 및 평가하기",
        type="primary"
    ):

        if not selected_model_features:

            st.error("독립 변수를 선택해 주세요.")

        elif data["_target"].value_counts().min() < 2:

            st.error("각 집단에 데이터가 최소 2개 이상 필요합니다.")

        else:

            X = data[selected_model_features].copy()
            y = data["_target"].copy()

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = [
                c for c in selected_model_features
                if c not in numeric_features
            ]

            transformers = []

            if numeric_features:

                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                transformers.append(
                    ("num", numeric_pipeline, numeric_features)
                )

            if categorical_features:

                categorical_pipeline = Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ])

                transformers.append(
                    ("cat", categorical_pipeline, categorical_features)
                )

            try:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size / 100,
                    random_state=int(random_seed),
                    stratify=y
                )

                models = {
                    "로지스틱 회귀": LogisticRegression(
                        max_iter=2000
                    ),

                    "의사결정트리": DecisionTreeClassifier(
                        max_depth=5,
                        random_state=int(random_seed)
                    )
                }

                results = {}
                saved_models = {}

                for name, estimator in models.items():

                    pipeline = Pipeline([
                        (
                            "preprocess",
                            ColumnTransformer(
                                transformers=transformers
                            )
                        ),
                        ("model", estimator)
                    ])

                    pipeline.fit(X_train, y_train)

                    predictions = pipeline.predict(X_test)

                    results[name] = {
                        "정확도": accuracy_score(
                            y_test, predictions
                        ),
                        "정밀도": precision_score(
                            y_test,
                            predictions,
                            zero_division=0
                        ),
                        "재현율": recall_score(
                            y_test,
                            predictions,
                            zero_division=0
                        ),
                        "F1 점수": f1_score(
                            y_test,
                            predictions,
                            zero_division=0
                        ),
                        "혼동행렬": confusion_matrix(
                            y_test,
                            predictions,
                            labels=[0, 1]
                        )
                    }

                    saved_models[name] = pipeline

                st.session_state["models"] = saved_models
                st.session_state["results"] = results
                st.session_state["features"] = selected_model_features

                st.success("모델 학습이 완료되었습니다.")

            except Exception as e:

                st.error(f"모델 학습 오류: {e}")

    if "results" in st.session_state:

        st.subheader("모델 성능 비교")

        results = st.session_state["results"]

        metric_table = pd.DataFrame({
            name: {
                metric: result[metric]
                for metric in [
                    "정확도",
                    "정밀도",
                    "재현율",
                    "F1 점수"
                ]
            }
            for name, result in results.items()
        })

        st.dataframe(
            metric_table.style.format("{:.3f}"),
            use_container_width=True
        )

        cols = st.columns(2)

        for i, (name, result) in enumerate(results.items()):

            with cols[i]:

                st.subheader(name)

                cm = result["혼동행렬"]

                cm_table = pd.DataFrame(
                    cm,
                    index=["실제 정상", "실제 당뇨병"],
                    columns=["정상 예측", "당뇨병 예측"]
                )

                st.dataframe(
                    cm_table,
                    use_container_width=True
                )

        st.info(
            "정확도: 전체 예측 중 정답을 맞힌 비율\n\n"
            "정밀도: 당뇨병이라고 예측한 사람 중 실제 당뇨병인 비율\n\n"
            "재현율: 실제 당뇨병 환자 중 올바르게 찾아낸 비율"
        )


# ==================================================
# 4. 직접 예측하기
# ==================================================

with tab4:

    st.header("4. 직접 당뇨병 발병 여부 예측하기")

    if "models" not in st.session_state:

        st.info(
            "먼저 분류 모델 탭에서 모델을 학습해 주세요."
        )

    else:

        features = st.session_state["features"]
        models = st.session_state["models"]

        st.write(
            "건강검진 값을 입력하면 "
            "두 모델이 당뇨병 발병 여부를 예측합니다."
        )

        input_values = {}

        for col in features:

            series = df[col]

            if pd.api.types.is_numeric_dtype(series):

                values = pd.to_numeric(
                    series,
                    errors="coerce"
                )

                median = values.median()

                if pd.isna(median):
                    median = 0.0

                min_value = values.min()
                max_value = values.max()

                if (
                    pd.isna(min_value)
                    or pd.isna(max_value)
                    or min_value == max_value
                ):
                    min_value = float(median) - 1
                    max_value = float(median) + 1

                input_values[col] = st.number_input(
                    col,
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(median),
                    key=f"input_{col}"
                )

            else:

                options = sorted(
                    series.dropna().astype(str).unique()
                )

                if not options:
                    options = ["정보 없음"]

                input_values[col] = st.selectbox(
                    col,
                    options,
                    key=f"input_{col}"
                )

        if st.button(
            "당뇨병 발병 여부 예측하기",
            type="primary"
        ):

            sample = pd.DataFrame(
                [input_values],
                columns=features
            )

            cols = st.columns(2)

            for i, (name, pipeline) in enumerate(models.items()):

                prediction = int(
                    pipeline.predict(sample)[0]
                )

                with cols[i]:

                    st.subheader(name)

                    if prediction == 1:

                        st.error("당뇨병으로 예측했습니다.")

                    else:

                        st.success("정상으로 예측했습니다.")

                    probability = pipeline.predict_proba(
                        sample
                    )[0][1]

                    st.metric(
                        "당뇨병 예측 확률",
                        f"{probability * 100:.1f}%"
                    )

        st.warning(
            "이 모델은 교육 및 실습용입니다. "
            "실제 의료 진단을 대신할 수 없습니다."
        )


# --------------------------------------------------
# 하단
# --------------------------------------------------

st.divider()

st.caption(
    "데이터 출처: Kaggle Diabetes Dataset. "
    "분석 결과는 제공된 데이터에 근거한 통계이며 "
    "의학적 진단 결과가 아닙니다."
)
