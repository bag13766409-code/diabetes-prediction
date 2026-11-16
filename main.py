import io
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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
st.caption("건강검진 데이터를 탐색하고 당뇨병 발병 여부를 예측하는 실습 앱입니다.")

KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "ehababoelnaga/diabetes-dataset"
)

ALIASES = {
    "나이": ["age", "나이"],
    "성별": ["gender", "sex", "성별"],
    "혈당": ["glucose", "blood glucose", "blood_glucose", "혈당"],
    "BMI": ["bmi", "체질량지수"],
    "혈압": ["bloodpressure", "blood pressure", "blood_pressure", "혈압"],
    "당뇨병 발병 여부": [
        "diabetes", "outcome", "class", "target",
        "diagnosis", "당뇨병", "당뇨 여부"
    ]
}


def normalize_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def find_column(columns, candidates):
    normalized = {normalize_name(c): c for c in columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    compact = {k.replace(" ", ""): v for k, v in normalized.items()}

    for candidate in candidates:
        key = normalize_name(candidate).replace(" ", "")
        if key in compact:
            return compact[key]

    return None


def read_csv_bytes(raw_bytes):
    for encoding in ("utf-8", "utf-8-sig", "cp949", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except Exception:
            pass

    raise ValueError("CSV 파일을 읽을 수 없습니다.")


@st.cache_data(ttl=3600, show_spinner=False)
def download_kaggle_dataset():
    response = requests.get(KAGGLE_DOWNLOAD_URL, timeout=35)
    response.raise_for_status()

    content = response.content

    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:

            csv_names = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
                and not name.startswith("__MACOSX")
            ]

            if not csv_names:
                raise ValueError("압축 파일에서 CSV를 찾지 못했습니다.")

            preferred = [
                name for name in csv_names
                if any(word in name.lower()
                       for word in ("training", "train", "diabetes"))
            ]

            chosen = preferred[0] if preferred else csv_names[0]

            with archive.open(chosen) as file:
                df = read_csv_bytes(file.read())

            return df, chosen

    return read_csv_bytes(content), "Kaggle CSV"


def prepare_binary_target(series):
    values = series.dropna().astype(str).str.strip()
    unique_values = list(pd.unique(values))

    if len(unique_values) != 2:
        raise ValueError(
            f"목표값은 두 종류여야 합니다. 현재 {len(unique_values)}종류입니다."
        )

    positive_words = [
        "1", "yes", "y", "true", "positive",
        "diabetes", "diabetic", "당뇨", "당뇨병", "발병"
    ]

    negative_words = [
        "0", "no", "n", "false", "negative",
        "normal", "non-diabetic", "non diabetic",
        "정상", "비당뇨"
    ]

    positive = None
    negative = None

    for value in unique_values:
        low = value.lower()

        if low in positive_words:
            positive = value

        if low in negative_words:
            negative = value

    if positive is not None and negative is not None:
        mapping = {
            negative: 0,
            positive: 1
        }

    else:
        try:
            numeric_values = sorted(float(v) for v in unique_values)

            if numeric_values == [0.0, 1.0]:
                mapping = {v: int(float(v)) for v in unique_values}
            else:
                ordered = sorted(unique_values)
                mapping = {
                    ordered[0]: 0,
                    ordered[1]: 1
                }

        except Exception:
            ordered = sorted(unique_values)
            mapping = {
                ordered[0]: 0,
                ordered[1]: 1
            }

    converted = series.astype(str).str.strip().map(mapping)

    return converted, mapping


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

st.sidebar.header("📂 데이터 준비")

st.sidebar.write(
    "Kaggle에서 데이터를 자동으로 불러옵니다. "
    "자동 다운로드가 안 되면 CSV 파일을 직접 올려 주세요."
)

uploaded_file = st.sidebar.file_uploader(
    "CSV 파일 직접 업로드",
    type=["csv"]
)

load_mode = st.sidebar.radio(
    "데이터 불러오기 방법",
    [
        "Kaggle에서 자동 불러오기",
        "CSV 파일 직접 업로드"
    ]
)

df = None
source_text = ""

if load_mode == "CSV 파일 직접 업로드":

    if uploaded_file is not None:
        try:
            df = read_csv_bytes(uploaded_file.getvalue())
            source_text = uploaded_file.name

        except Exception as e:
            st.error(f"파일을 읽는 중 문제가 발생했습니다: {e}")

else:

    with st.spinner("Kaggle 데이터를 불러오는 중입니다..."):

        try:
            df, source_text = download_kaggle_dataset()

        except Exception as e:

            st.warning(
                "Kaggle에서 자동으로 데이터를 가져오지 못했습니다. "
                "CSV 파일을 다운로드한 뒤 직접 업로드해 주세요."
            )

            st.caption(f"오류 내용: {e}")

            st.markdown(
                "[Kaggle 당뇨병 데이터셋 열기]"
                "(https://www.kaggle.com/datasets/ehababoelnaga/diabetes-dataset)"
            )

            if uploaded_file is not None:
                try:
                    df = read_csv_bytes(uploaded_file.getvalue())
                    source_text = uploaded_file.name

                except Exception as upload_error:
                    st.error(f"업로드 오류: {upload_error}")


if df is None:
    st.info("데이터를 불러오면 분석 화면이 나타납니다.")
    st.stop()


df = df.copy()
df.columns = [str(c).strip() for c in df.columns]
df = df.dropna(how="all")

st.sidebar.success(f"데이터 불러오기 완료: {source_text}")
st.sidebar.caption(
    f"{len(df):,}행 × {len(df.columns):,}열"
)


# 주요 컬럼 자동 탐색

detected = {
    label: find_column(df.columns, names)
    for label, names in ALIASES.items()
}


# 목표 컬럼 선택

st.sidebar.divider()
st.sidebar.subheader("분석에 사용할 컬럼")

target_candidates = list(df.columns)

default_target = detected["당뇨병 발병 여부"]

target_index = (
    target_candidates.index(default_target)
    if default_target in target_candidates
    else 0
)

target_col = st.sidebar.selectbox(
    "당뇨병 발병 여부 컬럼(정답)",
    target_candidates,
    index=target_index
)


# --------------------------------------------------
# 화면 구성
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

    st.write(
        "데이터의 크기, 컬럼 이름, 자료형, 빈 값의 개수를 확인합니다."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("전체 데이터 수", f"{len(df):,}명")
    c2.metric("속성(컬럼) 수", f"{len(df.columns):,}개")
    c3.metric(
        "빈 값이 있는 셀",
        f"{int(df.isna().sum().sum()):,}개"
    )

    st.subheader("데이터 미리보기")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("컬럼별 정보")

    meaning_map = {
        "나이": "건강검진을 받은 사람의 나이",
        "성별": "건강검진을 받은 사람의 성별",
        "혈당": "혈액 속 포도당의 농도를 나타내는 수치",
        "BMI": "키와 몸무게를 이용해 계산한 비만 정도",
        "혈압": "혈액이 혈관 벽에 가하는 압력",
        "당뇨병 발병 여부": "당뇨병 발병 여부(1: 당뇨병, 0: 정상)"
    }

    description_rows = []

    for col in df.columns:

        role = next(
            (label for label, found in detected.items() if found == col),
            ""
        )

        description_rows.append({
            "컬럼명": col,
            "우리말 뜻": meaning_map.get(role, ""),
            "데이터 종류": str(df[col].dtype),
            "빈 값 개수": int(df[col].isna().sum()),
            "고유값 개수": int(df[col].nunique(dropna=True))
        })

    st.dataframe(
        pd.DataFrame(description_rows),
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
        "당뇨병 발병 여부에 따라 건강 지표의 분포가 "
        "어떻게 달라지는지 살펴봅니다."
    )

    try:
        target_numeric, target_mapping = prepare_binary_target(
            df[target_col]
        )

    except Exception as e:
        st.error(f"목표 컬럼을 확인해 주세요: {e}")
        st.stop()

    valid_target = target_numeric.notna()

    explore_df = df.loc[valid_target].copy()

    explore_df["_당뇨병_숫자"] = (
        target_numeric.loc[valid_target].astype(int)
    )

    positive_count = int(
        (explore_df["_당뇨병_숫자"] == 1).sum()
    )

    negative_count = int(
        (explore_df["_당뇨병_숫자"] == 0).sum()
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("정상(0)", f"{negative_count:,}명")
    m2.metric("당뇨병(1)", f"{positive_count:,}명")

    ratio = (
        positive_count / len(explore_df) * 100
        if len(explore_df) > 0
        else 0
    )

    m3.metric("당뇨병 비율", f"{ratio:.1f}%")

    st.caption(
        f"목표값 변환 기준: {target_mapping} "
        "(0=정상, 1=당뇨병)"
    )

    numeric_cols = explore_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_cols = [
        c for c in numeric_cols
        if c != "_당뇨병_숫자"
    ]

    if numeric_cols:

        preferred_order = [
            detected["혈당"],
            detected["BMI"],
            detected["나이"],
            detected["혈압"]
        ]

        preferred_order = [
            c for c in preferred_order
            if c in numeric_cols
        ]

        remaining = [
            c for c in numeric_cols
            if c not in preferred_order
        ]

        chart_cols = preferred_order + remaining

        selected_feature = st.selectbox(
            "분포를 확인할 숫자형 속성",
            chart_cols
        )

        st.subheader(
            f"정상과 당뇨병 그룹의 {selected_feature} 비교"
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        for label in [0, 1]:

            values = pd.to_numeric(
                explore_df.loc[
                    explore_df["_당뇨병_숫자"] == label,
                    selected_feature
                ],
                errors="coerce"
            ).dropna()

            if len(values):

                ax.hist(
                    values,
                    bins=25,
                    alpha=0.55,
                    label="정상" if label == 0 else "당뇨병"
                )

        ax.set_xlabel(selected_feature)
        ax.set_ylabel("인원수")
        ax.set_title(
            f"정상과 당뇨병 그룹의 {selected_feature} 분포"
        )
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("당뇨병 여부에 따른 평균 비교")

        means = explore_df.groupby(
            "_당뇨병_숫자"
        )[numeric_cols].mean().T

        means = means.rename(
            columns={
                0: "정상 평균",
                1: "당뇨병 평균"
            }
        )

        st.dataframe(
            means,
            use_container_width=True
        )

        if len(numeric_cols) >= 2:

            st.subheader("숫자형 속성 사이의 상관관계")

            corr = explore_df[
                numeric_cols + ["_당뇨병_숫자"]
            ].corr(numeric_only=True)

            fig, ax = plt.subplots(figsize=(9, 6))

            image = ax.imshow(
                corr,
                cmap="coolwarm",
                vmin=-1,
                vmax=1
            )

            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(
                corr.columns,
                rotation=45,
                ha="right"
            )

            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)

            fig.colorbar(image, ax=ax)
            ax.set_title("상관관계")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("탐색할 숫자형 컬럼이 없습니다.")


# ==================================================
# 3. 분류 모델
# ==================================================

with tab3:

    st.header("3. 분류 모델 만들기")

    st.write(
        "데이터를 학습용과 시험용으로 나눈 뒤 "
        "두 가지 분류 모델을 학습합니다."
    )

    st.caption(
        "당뇨병(1)을 양성으로 설정합니다."
    )

    model_df = df.copy()

    try:
        y_converted, mapping = prepare_binary_target(
            model_df[target_col]
        )

    except Exception as e:
        st.error(f"목표 컬럼 오류: {e}")
        st.stop()

    model_df["_target"] = y_converted

    model_df = model_df.dropna(
        subset=["_target"]
    )

    feature_options = [
        c for c in model_df.columns
        if c not in [target_col, "_target"]
    ]

    default_features = []

    for key in ["나이", "성별", "혈당", "BMI", "혈압"]:

        col = detected.get(key)

        if (
            col in feature_options
            and col not in default_features
        ):
            default_features.append(col)

    if not default_features:
        default_features = feature_options[:5]

    selected_features = st.multiselect(
        "독립 변수로 사용할 속성을 선택하세요.",
        feature_options,
        default=default_features
    )

    test_size = st.slider(
        "시험용 데이터 비율(%)",
        10,
        40,
        20,
        5
    )

    random_seed = st.number_input(
        "데이터 나누기 기준 숫자",
        min_value=0,
        max_value=9999,
        value=42
    )

    if st.button(
        "모델 학습하고 평가하기",
        type="primary"
    ):

        if len(selected_features) == 0:

            st.error("독립 변수를 한 개 이상 선택해 주세요.")

        elif model_df["_target"].nunique() != 2:

            st.error(
                "목표 컬럼에 정상과 당뇨병, "
                "두 종류의 값이 모두 있어야 합니다."
            )

        elif model_df["_target"].value_counts().min() < 2:

            st.error(
                "각 목표값의 데이터가 최소 2개 이상 있어야 합니다."
            )

        else:

            X = model_df[selected_features].copy()
            y = model_df["_target"].astype(int)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = [
                c for c in selected_features
                if c not in numeric_features
            ]

            numeric_pipeline = Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ])

            categorical_pipeline = Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ])

            transformers = []

            if numeric_features:
                transformers.append(
                    ("num", numeric_pipeline, numeric_features)
                )

            if categorical_features:
                transformers.append(
                    ("cat", categorical_pipeline, categorical_features)
                )

            preprocessor = ColumnTransformer(
                transformers=transformers
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
                fitted_pipelines = {}

                for model_name, estimator in models.items():

                    pipeline = Pipeline(steps=[
                        (
                            "preprocess",
                            ColumnTransformer(
                                transformers=transformers
                            )
                        ),
                        (
                            "model",
                            estimator
                        )
                    ])

                    pipeline.fit(X_train, y_train)

                    predictions = pipeline.predict(X_test)

                    results[model_name] = {
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

                    fitted_pipelines[model_name] = pipeline

                st.session_state["diabetes_model_results"] = results
                st.session_state["diabetes_models"] = fitted_pipelines
                st.session_state["diabetes_features"] = selected_features
                st.session_state["diabetes_test_y"] = y_test

                st.success("두 모델의 학습과 평가가 끝났습니다.")

            except Exception as e:

                st.error(
                    f"모델을 학습하는 중 문제가 발생했습니다: {e}"
                )

    if "diabetes_model_results" in st.session_state:

        results = st.session_state["diabetes_model_results"]

        st.subheader("모델 성능 비교")

        metric_table = pd.DataFrame({
            name: {
                metric: values[metric]
                for metric in [
                    "정확도",
                    "정밀도",
                    "재현율",
                    "F1 점수"
                ]
            }
            for name, values in results.items()
        })

        st.dataframe(
            metric_table.style.format("{:.3f}"),
            use_container_width=True
        )

        cols = st.columns(2)

        for i, (name, values) in enumerate(results.items()):

            with cols[i]:

                st.subheader(name)

                cm = values["혼동행렬"]

                cm_df = pd.DataFrame(
                    cm,
                    index=["실제 정상", "실제 당뇨병"],
                    columns=["정상 예측", "당뇨병 예측"]
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                st.caption(
                    f"정확도 {values['정확도']:.3f} · "
                    f"정밀도 {values['정밀도']:.3f} · "
                    f"재현율 {values['재현율']:.3f}"
                )

        st.info(
            "정확도: 전체 예측 중 실제 정답을 맞힌 비율\n\n"
            "정밀도: 당뇨병이라고 예측한 사람 중 실제 당뇨병인 비율\n\n"
            "재현율: 실제 당뇨병 환자 중 당뇨병이라고 찾아낸 비율"
        )


# ==================================================
# 4. 직접 예측하기
# ==================================================

with tab4:

    st.header("4. 새로운 값으로 예측해 보기")

    if "diabetes_models" not in st.session_state:

        st.info(
            "먼저 '분류 모델' 탭에서 모델을 학습해 주세요."
        )

    else:

        features = st.session_state["diabetes_features"]
        models = st.session_state["diabetes_models"]

        st.write(
            "아래에 건강검진 값을 입력하면 "
            "학습한 모델의 예측 결과를 확인할 수 있습니다."
        )

        input_values = {}

        for col in features:

            series = df[col]

            if pd.api.types.is_numeric_dtype(series):

                numeric_series = pd.to_numeric(
                    series,
                    errors="coerce"
                )

                median = numeric_series.median()

                if pd.isna(median):
                    median = 0.0

                min_val = numeric_series.min()
                max_val = numeric_series.max()

                if (
                    pd.isna(min_val)
                    or pd.isna(max_val)
                    or min_val == max_val
                ):
                    min_val = float(median) - 1
                    max_val = float(median) + 1

                input_values[col] = st.number_input(
                    col,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(median),
                    key=f"prediction_{col}"
                )

            else:

                options = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if not options:
                    options = ["알 수 없음"]

                input_values[col] = st.selectbox(
                    col,
                    options,
                    key=f"prediction_{col}"
                )

        if st.button(
            "당뇨병 발병 여부 예측",
            type="primary"
        ):

            sample = pd.DataFrame(
                [input_values],
                columns=features
            )

            prediction_cols = st.columns(2)

            for i, (name, pipeline) in enumerate(models.items()):

                prediction = int(
                    pipeline.predict(sample)[0]
                )

                with prediction_cols[i]:

                    st.subheader(name)

                    if prediction == 1:
                        st.error("당뇨병으로 예측했습니다.")

                    else:
                        st.success("정상으로 예측했습니다.")

                    if hasattr(pipeline, "predict_proba"):

                        probability = pipeline.predict_proba(
                            sample
                        )[0][1]

                        st.metric(
                            "당뇨병으로 예측한 확률",
                            f"{probability * 100:.1f}%"
                        )

        st.warning(
            "이 결과는 교육 및 실습용 예측이며, "
            "의학적 진단이나 검사 결과를 대신할 수 없습니다. "
            "건강 상태가 걱정된다면 의료 전문가와 상담하세요."
        )


# --------------------------------------------------
# 하단 안내
# --------------------------------------------------

st.divider()

st.caption(
    "데이터 출처: Kaggle - Diabetes Dataset "
    "(Ehab Aboelnaga). "
    "실제 CSV의 컬럼 구성을 확인해 주세요."
)
