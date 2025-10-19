import os
import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def auto_detect_data_path():
    candidates = [
        "최종_데이터.csv",
        "학원 데이터_정리_v6_컬럼정리__표시명추가.csv",
        "진짜 최종본 자료.xlsx",
        "academies_labeled.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            print("✅ 자동으로 데이터 파일을 찾았습니다:", c)
            return c
    raise FileNotFoundError("❌ 데이터 파일을 찾을 수 없습니다. CSV 또는 XLSX 파일을 같은 폴더에 두세요.")

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype=object)
    elif ext in [".csv", ".txt"]:
        for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]:
            try:
                return pd.read_csv(path, dtype=object, encoding=enc, engine="python")
            except Exception:
                pass
        raise RuntimeError("CSV 인코딩을 확인하세요.")
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. (csv, xlsx 권장)")

def detect_admin_col(df: pd.DataFrame):
    exact_priority = ["행정구역명", "시군구", "시/군/구", "읍면동", "행정동", "자치구"]
    for c in df.columns:
        if str(c) in exact_priority:
            return c
    keywords = ["행정구역", "시군구", "시/군/구", "읍면동", "행정동", "자치구"]
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in [k.lower() for k in keywords]):
            if "구분" in str(c):
                continue
            return c
    for c in df.columns:
        if any(k in str(c) for k in ["주소", "address"]):
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]

def detect_name_col(df: pd.DataFrame):
    priority = ["표시명", "학원명_추가", "학원명", "학원이름", "기관명", "academy", "name"]
    for p in priority:
        for c in df.columns:
            if str(c).lower() == p.lower():
                return c
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["표시명", "학원명", "학원이름", "기관명", "academy", "name"]):
            return c
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        uniq_counts = [(c, df[c].astype(str).nunique(dropna=True)) for c in obj_cols]
        uniq_counts.sort(key=lambda x: x[1], reverse=True)
        return uniq_counts[0][0]
    return df.columns[0]

def ensure_display_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "표시명" in df.columns:
        return df
    admin_col = detect_admin_col(df)
    name_col  = detect_name_col(df)
    base = df[admin_col].astype(str).fillna("")
    name = df[name_col].astype(str).fillna("")
    disp = (base.str.strip() + " " + name.str.strip()).str.strip()
    df["표시명"] = disp.replace({"^$": np.nan}, regex=True)
    return df

def split_feature_columns(df: pd.DataFrame, exclude_cols):
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c in exclude_cols:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().mean() >= 0.5:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    for c in categorical_cols[:]:
        if any(k in str(c).lower() for k in ["평점","rating","점수","리뷰","수강","학생수","정원","비용","가격","fee"]):
            categorical_cols.remove(c)
            numeric_cols.append(c)
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
    except TypeError:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocess

def fit_knn(X_trans, metric="euclidean", n_neighbors=10):
    return NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X_trans)

def apply_hard_filters(df: pd.DataFrame, query_dict: dict, exclude_cols):
    if not query_dict:
        return df
    df_f = df.copy()
    for k, v in query_dict.items():
        if k not in df_f.columns:
            continue
        s = df_f[k]
        num_mask = pd.to_numeric(s, errors="coerce").notna()
        if pd.to_numeric(pd.Series([v]), errors="coerce").notna().iloc[0] and num_mask.mean() >= 0.5:
            vs = pd.to_numeric(s, errors="coerce")
            vv = float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
            df_f = df_f[np.isfinite(vs) & (np.abs(vs - vv) <= 1e-6)]
        else:
            def norm(x): return str(x).strip().lower()
            nv = norm(v)
            df_f = df_f[s.apply(lambda x: norm(x) == nv)]
        if len(df_f) == 0:
            break
    return df_f

def recommend(df, preprocess, knn, exclude_cols, query_dict, top_n=5):
    X = df.drop(columns=list(exclude_cols), errors="ignore")
    q = pd.DataFrame([{c: np.nan for c in X.columns}])
    for k, v in (query_dict or {}).items():
        if k in q.columns:
            q.at[0, k] = v
    Q = preprocess.transform(q)
    n_nbrs = min(top_n, len(df))
    distances, indices = knn.kneighbors(Q, n_neighbors=n_nbrs)
    idx = indices[0].tolist()
    dist = distances[0].tolist()
    result = df.iloc[idx].copy()
    result.insert(0, "similarity_distance", dist)
    return result.head(top_n)

DATA_PATH = auto_detect_data_path()
df_all = read_any(DATA_PATH)
df_all = ensure_display_name(df_all)

cols = [str(c) for c in df_all.columns]
name_col = detect_name_col(df_all)
admin_col = detect_admin_col(df_all)
exclude_keywords = ["표시명","행정구역명", "과목명", "학원명","학원이름","기관명","name","academy","전화","연락처","주소","위치","좌표","url","홈페이지","메모","비고","설명"]
exclude_cols = set([c for c in cols if any(k in str(c).lower() for k in exclude_keywords)])
numeric_cols, categorical_cols = split_feature_columns(df_all, exclude_cols)
DF_STATE = df_all.copy()
PREPROCESS = None
KNN = None

def prepare_model(admin_filter, metric = "euclidean", topn_build=25):
    global DF_STATE, PREPROCESS, KNN, numeric_cols, categorical_cols
    df = df_all.copy()
    if admin_filter and admin_filter != "(전체)" and admin_col in df.columns:
        df = df[df[admin_col].astype(str) == str(admin_filter)].copy()
    X = df.drop(columns=list(exclude_cols), errors="ignore")
    num_cols, cat_cols = split_feature_columns(df, exclude_cols)
    PREPROCESS = build_preprocessor(num_cols, cat_cols)
    PREPROCESS.fit(X)
    X_trans = PREPROCESS.transform(X)
    KNN = fit_knn(X_trans, metric=metric, n_neighbors=min(topn_build, len(df)))
    DF_STATE = df
    numeric_cols[:] = num_cols
    categorical_cols[:] = cat_cols
    info = {"행 수": len(df), "학원명 컬럼": str(name_col), "행정구역 컬럼": str(admin_col), "숫자형 특성 수": len(num_cols), "범주형 특성 수": len(cat_cols), "거리(metric)": metric}
    return json.dumps(info, ensure_ascii=False, indent=2)

def run_recommend_form(*args):
    if PREPROCESS is None or KNN is None:
        return "먼저 상단에서 모델을 준비하세요.", None, None
    metric = args[-2]
    top_n = int(args[-1])
    X_all = DF_STATE.drop(columns=list(exclude_cols), errors="ignore")
    feat_cols = list(X_all.columns)
    values = list(args[:-2])
    query_dict = {}
    for c, v in zip(feat_cols, values):
        if v is None or (isinstance(v, str) and v.strip() in ["", "(선택 안함)"]):
            continue
        query_dict[c] = v
    df_f = apply_hard_filters(DF_STATE, query_dict, exclude_cols)
    if len(df_f) == 0:
        return "선택한 조건을 모두 만족하는 학원이 없습니다. 조건을 완화해보세요.", None, None
    Xf = df_f.drop(columns=list(exclude_cols), errors="ignore")
    Qf = PREPROCESS.transform(Xf)
    knn_local = fit_knn(Qf, metric=metric, n_neighbors=min(max(10, top_n), len(df_f)))
    result = recommend(df_f, PREPROCESS, knn_local, exclude_cols, query_dict, top_n)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    result.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    return "완료", result, tmp.name

def run_recommend_form(*args):
    if PREPROCESS is None or KNN is None:
        return "먼저 상단에서 모델을 준비하세요.", None
    metric = "euclidean"
    top_n = int(args[-1])
    X_all = DF_STATE.drop(columns=list(exclude_cols), errors="ignore")
    feat_cols = list(X_all.columns)
    values = list(args[:-1])
    query_dict = {}
    for c, v in zip(feat_cols, values):
        if v is None or (isinstance(v, str) and v.strip() in ["", "(선택 안함)"]):
            continue
        query_dict[c] = v

    df_f = apply_hard_filters(DF_STATE, query_dict, exclude_cols)
    if len(df_f) == 0:
        return "선택한 조건을 모두 만족하는 학원이 없습니다. 조건을 완화해보세요.", None

    Xf = df_f.drop(columns=list(exclude_cols), errors="ignore")
    Qf = PREPROCESS.transform(Xf)
    knn_local = fit_knn(Qf, metric=metric, n_neighbors=min(max(10, top_n), len(df_f)))
    result = recommend(df_f, PREPROCESS, knn_local, exclude_cols, query_dict, top_n)
    return "완료", result

custom_theme = gr.themes.Base(primary_hue="pink", secondary_hue="pink").set(
    body_background_fill="#ffffff",
    background_fill_primary="#ffffff",
    background_fill_secondary="#ffe6f0",
    border_color_primary="#ffc0cb",
    button_primary_background_fill="#ffc0cb",
    button_primary_background_fill_hover="#ffb6c1",
    button_primary_text_color="#4a001f",
    block_title_text_color="#d63384"
)

with gr.Blocks(title="학원 추천기 (KNN·하드필터)", theme=custom_theme) as demo:
    gr.HTML(
        '''
        <div style="display:flex; align-items:center; justify-content:center; gap:20px; margin-top:30px;">
            
            <!-- 🟪 왼쪽 해커톤 로고 -->
            <img src="https://drive.google.com/thumbnail?id=1WlOO7svUbfCE0fVOfkQWht4en2Rqjs3P"
                 alt="해커톤 로고"
                 style="width:70px; height:70px; border-radius:15px; box-shadow:0px 3px 6px rgba(0,0,0,0.15);">

            <!-- 🎓 Instudygram 로고 (Postimg 이미지) -->
            <div style="background:#fff; padding:12px 25px; border-radius:15px; box-shadow:0px 4px 8px rgba(0,0,0,0.08);">
                <img src="https://i.postimg.cc/Pxh4JXLW/image.png"
                     alt="Instudygram 로고"
                     style="width:200px; height:auto; display:block;">
            </div>

        </div>

        <div style="text-align:center; margin-top:10px;">
            <p style="color:#888; font-size:18px; margin-top:5px;">
                AI 기반 학원 추천 플랫폼
            </p>
        </div>
        '''
    )



    with gr.Row():
        if admin_col in df_all.columns:
            admin_choices = ["(전체)"] + sorted([str(v) for v in df_all[admin_col].dropna().astype(str).unique()])
        else:
            admin_choices = ["(전체)"]
        admin_filter = gr.Dropdown(
        choices=admin_choices,
        value="(전체)",
        label="지역 필터",
        scale=1,
    )

    prepare_btn = gr.Button("검색🔍", variant="primary", scale=1)
    prep_info = gr.Code(label="모델 요약", interactive=False, elem_id="prep-box")
    

    # metric 제거된 상태로 변경
    prepare_btn.click(prepare_model, inputs=[admin_filter], outputs=[prep_info])


    gr.Markdown("<br><br>", visible=False)

    with gr.Tab("폼 입력"):
        X = df_all.drop(columns=list(exclude_cols), errors="ignore")
        feat_cols = list(X.columns)
        inputs = []
        with gr.Accordion("입력 폼(미선택 시에는 지역 조건만 적용됩니다. )", open=True):
            for c in feat_cols:
                ser = DF_STATE[c].dropna()
                if pd.to_numeric(ser, errors="coerce").notna().mean() >= 0.5:
                    comp = gr.Number(label=f"{c} (숫자)")
                else:
                    choices = ["(선택 안함)"] + sorted([str(x) for x in ser.astype(str).unique()][:200])
                    comp = gr.Dropdown(choices=choices, value="(선택 안함)", label=f"{c}")
                inputs.append(comp)
        with gr.Row():
            top_n = gr.Slider(3, 20, value=5, step=1, label="추천 개수 Top-N")
            run_btn = gr.Button("🔎 추천 실행", variant="primary", elem_id = "run-btn")
        status1 = gr.Textbox(label="상태", interactive=False)
        out_df1 = gr.Dataframe(label="추천 결과", interactive=False, wrap=True)
        run_btn.click(run_recommend_form, inputs=[*inputs, top_n], outputs=[status1, out_df1])

    

demo.css = """
/* ================================
   📌 전체 레이아웃 및 기본 설정
================================ */
#filter-row {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 25px;
    margin-top: 5px;
    margin-bottom: 20px;
}

#filter-box, #metric-box {
    background-color: #ffffff; /* 내부 흰색 */
    border: 2px solid #ffb6c1; /* 윤곽선만 분홍색 */
    border-radius: 25px;
    padding: 25px;
    width: 260px;
    height: 130px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(255, 182, 193, 0.25);
    transition: all 0.2s ease-in-out;
}

#filter-box:hover, #metric-box:hover {
    transform: scale(1.03);
    box-shadow: 0px 6px 14px rgba(255, 182, 193, 0.4);
}

/* ================================
   🎀 전체 컨테이너 둥근 스타일
================================ */
.gradio-container, .block, .box, .accordion, .dataframe {
    border-radius: 25px !important;
    overflow: hidden;
}

/* ================================
   💗 버튼 스타일
================================ */
.gr-button-primary {
    background-color: #ffb6c1 !important;
    border: none !important;
    color: #4a001f !important;
    border-radius: 20px !important;
    font-weight: 600;
}
.gr-button-primary:hover {
    background-color: #ff99cc !important;
}

/* ================================
   ✨ 입력창 및 폼 스타일
================================ */
.input-field, .gr-input, .gr-textbox, .gr-dropdown, .gr-number, .gr-slider {
    font-size: 18px !important;
    color: #4a001f !important;
    background-color: #ffffff !important; /* 내부 흰색 */
    border: 1.5px solid #ffb6c1 !important;
    border-radius: 18px !important;
    padding: 10px !important;
    box-shadow: 0 3px 8px rgba(255, 182, 193, 0.2);
}

/* ================================
   🩰 폼 전체 카드 스타일
================================ */
.accordion, .block, .box {
    background-color: #ffffff !important; /* 박스 내부 흰색 */
    border: 2px solid #ffb6c1 !important; /* 분홍 윤곽선 */
    border-radius: 25px !important;
    padding: 25px !important;
    box-shadow: 0px 4px 10px rgba(255, 182, 193, 0.25);
}

/* 내부 입력 요소 경계 제거 */
.accordion > .block, .accordion > div > .block {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 8px 0 !important;
}

/* 폼 타이틀 */
.accordion label {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #d63384 !important;
}

/* 로고 중앙 정렬 */
#logo-header {
    text-align: center;
    margin-top: 20px;
}
#logo-header img {
    width: 260px;
    height: auto;
    margin-bottom: 10px;
    filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.1));
}
#logo-header p {
    color: #888;
    font-size: 18px;
    margin-top: 5px;
}

/* 전체 여백 조정 */
#filter-row { margin-top: 5px !important; }
.accordion { margin-top: -10px !important; }

.tab-nav, .tab, .form {
    border-radius: 20px !important;
}

/* 결과 테이블 */
.dataframe {
    background-color: #fff;
    border: 1.5px solid #ffb6c1 !important;
    border-radius: 20px !important;
    box-shadow: 0 3px 8px rgba(255, 182, 193, 0.2);
}

#run-btn {
    width: 200px !important;
    height: 110px !important;
    font-size: 20px !important;
}
"""
demo.launch(debug=False, share=True)
