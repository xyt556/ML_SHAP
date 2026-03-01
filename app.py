# -*- coding: utf-8 -*-
"""
Streamlit 应用：机器学习与 SHAP 可解释性分析
支持分类/回归、多种模型、SHAP 摘要图与依赖图等。
"""

import io
import zipfile
import hashlib
import pickle
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 字体配置：优先使用 Linux 中文字体（Streamlit Cloud），再回退到 Windows 字体
# packages.txt 需包含 fonts-noto-cjk，否则 Linux 下中文会显示为方框
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",  # Linux (fonts-noto-cjk)
    "Microsoft YaHei", "微软雅黑", "SimHei",                      # Windows
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
# PDF/EPS 使用 TrueType 字体（42），避免 Type 3 字体导致编辑软件无法编辑文字
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import xgboost
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="ML & SHAP 分析 - @3S&ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式（含默认字体、暗色主题）
def _get_theme_css(dark):
    base = """
    html, body, [class*="css"] { font-family: 'Microsoft YaHei', '微软雅黑', sans-serif !important; }
    .main-header { font-size: 2.8rem; font-weight: 700; margin-bottom: 1rem; }
    .metric-card { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    div[data-testid="stSidebar"] .stMarkdown { font-size: 0.95rem; }
    """
    if dark:
        base += """
        .stApp { background-color: #0e1117; }
        .main-header { color: #4da6ff; }
        .metric-card { background: #1e2130; }
        [data-testid="stMetricValue"] { color: #fafafa; }
        .stDataFrame { background: #1e2130; }
        """
    else:
        base += """
        .main-header { color: #1f77b4; }
        .metric-card { background: #f0f2f6; }
        """
    return base


@st.cache_data
def load_sample_classification():
    """加载分类示例数据（鸢尾花风格）。"""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6, n_redundant=2,
        n_classes=3, random_state=42
    )
    feature_names = [f"特征_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["目标"] = y
    return df


@st.cache_data
def load_sample_regression():
    """加载回归示例数据。"""
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=500, n_features=10, n_informative=6, noise=10, random_state=42
    )
    feature_names = [f"特征_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["目标"] = y
    return df


def load_data(uploaded_file):
    """根据上传文件类型读取数据。"""
    if uploaded_file is None:
        return None
    suffix = (uploaded_file.name or "").lower()
    if suffix.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if suffix.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    st.error("仅支持 CSV 或 Excel 文件")
    return None


def prepare_X_y(df, target_col, problem_type, feat_cols=None):
    """准备特征 X 与目标 y，并对分类目标编码。feat_cols 为可选的特征列列表。"""
    if feat_cols is not None and len(feat_cols) > 0:
        X = df[feat_cols].copy()
    else:
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    le = None
    if problem_type == "分类" and not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    return X, y, le


def _get_all_models(problem_type):
    """返回所有可用模型（含可选依赖）。"""
    if problem_type == "分类":
        models = {
            "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
            "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
            "梯度提升": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgboost.XGBClassifier(n_estimators=100, random_state=42)
        if LGB_AVAILABLE:
            models["LightGBM"] = lightgbm.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        models = {
            "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
            "岭回归": Ridge(alpha=1.0, random_state=42),
            "梯度提升": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "SVM": SVR(kernel="rbf"),
            "KNN": KNeighborsRegressor(n_neighbors=5),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgboost.XGBRegressor(n_estimators=100, random_state=42)
        if LGB_AVAILABLE:
            models["LightGBM"] = lightgbm.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    return models


def get_model(problem_type, model_name):
    """根据问题类型和模型名称返回 sklearn 模型。"""
    models = _get_all_models(problem_type)
    return models.get(model_name, list(models.values())[0])


def _is_tree_model(model):
    """判断是否为树模型（支持 TreeExplainer）。"""
    if hasattr(model, "estimators_"):
        return True
    if XGB_AVAILABLE and "xgboost" in str(type(model).__module__):
        return True
    if LGB_AVAILABLE and "lightgbm" in str(type(model).__module__):
        return True
    return False


def run_shap_analysis(model, X_train, X_explain, problem_type):
    """运行 SHAP 分析并返回 explainer 与 values（原始结构，含多分类的 list/3D）。"""
    if not SHAP_AVAILABLE:
        return None, None, None
    try:
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model, X_train)
            shap_values = explainer.shap_values(X_explain, check_additivity=False)
        else:
            try:
                masker = shap.maskers.Independent(X_train)
                explainer = shap.LinearExplainer(model, masker)
            except (TypeError, AttributeError):
                explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_explain)
        return explainer, shap_values, X_explain
    except Exception as e:
        st.warning(f"SHAP 解释器创建失败: {e}")
        return None, None, None


def _shap_to_2d(shap_values, class_idx=0):
    """
    将多分类 SHAP 值统一为 2D (n_samples, n_features)，供 summary/dependence 使用。
    - list: 取 shap_values[class_idx]
    - 3D (n_samples, n_features, n_classes): 取 [:, :, class_idx]
    - 2D: 原样返回
    """
    sv = np.asarray(shap_values)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx])
    if sv.ndim == 3:
        return sv[:, :, class_idx]
    return sv


# 图形格式与 MIME 类型映射
FIG_FORMAT_MAP = {
    "PNG": ("png", "image/png"),
    "PDF": ("pdf", "application/pdf"),
    "SVG": ("svg", "image/svg+xml"),
    "EPS": ("eps", "application/postscript"),
}


def save_fig_bytes(fig, fmt, plot_name="shap_plot", dpi=150):
    """
    将 matplotlib 图形保存为字节流，支持 PNG/PDF/SVG/EPS。
    PDF/SVG/EPS 为矢量格式，可在 Illustrator/Inkscape 等软件中编辑。
    """
    fmt_upper = (fmt or "PNG").strip().upper()
    ext, _ = FIG_FORMAT_MAP.get(fmt_upper, FIG_FORMAT_MAP["PNG"])
    buf = io.BytesIO()
    if ext == "pdf":
        fig.savefig(buf, format="pdf", bbox_inches="tight", metadata={"Creator": "Streamlit ML-SHAP"})
    elif ext == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
    elif ext == "eps":
        fig.savefig(buf, format="eps", bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        ext = "png"
    buf.seek(0)
    return buf.getvalue(), f"{plot_name}.{ext}"


def main():
    st.markdown('<p class="main-header">📊 机器学习与 SHAP 可解释性分析 @3S&ML</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("界面与体验")
        dark_theme = st.checkbox("🌙 暗色主题", key="dark_theme")
        use_cache = st.checkbox("启用数据缓存", value=True, help="缓存训练与 SHAP 结果，切换参数时加快响应")
        st.markdown("---")
        st.header("数据与模型")
        data_source = st.radio("数据来源", ["上传文件", "示例数据（分类）", "示例数据（回归）"])
        problem_type = st.selectbox("任务类型", ["分类", "回归"])
        all_models = _get_all_models(problem_type)
        model_names = list(all_models.keys())
        compare_mode = st.checkbox("模型对比（训练多个模型）", value=False)
        if compare_mode:
            model_selected = st.multiselect("选择要对比的模型", model_names, default=model_names[:3])
        else:
            model_name = st.selectbox("模型", model_names)
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        use_scaling = st.checkbox("对特征做标准化", value=False)
        enable_tuning = st.checkbox("超参数调优", value=False)
        tune_method = "网格搜索 (GridSearchCV)"
        tune_cv = 5
        if enable_tuning:
            tune_method = st.radio("调优方式", ["网格搜索 (GridSearchCV)", "随机搜索 (RandomizedSearchCV)"], horizontal=True)
            tune_cv = st.slider("调优交叉验证折数", 2, 10, 5)
        enable_cv = st.checkbox("K 折交叉验证", value=False)
        cv_folds = 5
        if enable_cv:
            cv_folds = st.slider("交叉验证折数 K", 2, 10, 5)
    st.markdown(f"<style>{_get_theme_css(dark_theme)}</style>", unsafe_allow_html=True)

    df = None
    if data_source == "上传文件":
        uploaded = st.file_uploader("上传 CSV 或 Excel", type=["csv", "xlsx", "xls"])
        df = load_data(uploaded)
        if df is not None and df.empty:
            st.warning("表格为空")
            return
    elif data_source == "示例数据（分类）":
        df = load_sample_classification()
        problem_type = "分类"
    else:
        df = load_sample_regression()
        problem_type = "回归"

    if df is None:
        st.info("请选择数据来源并上传文件或使用示例数据。")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("未检测到数值列。")
        return

    target_col = st.selectbox(
        "选择目标列",
        df.columns.tolist(),
        index=min(len(df.columns) - 1, list(df.columns).index(numeric_cols[-1]) if numeric_cols[-1] in df.columns else 0)
    )

    # 数据探索与预处理
    st.subheader("数据探索与预处理")
    df_key = f"df_processed_{id(df)}"
    if df_key not in st.session_state:
        st.session_state[df_key] = df.copy()
    df_work = st.session_state[df_key]
    feat_cols = [c for c in df_work.columns if c != target_col and c in numeric_cols]
    if not feat_cols:
        feat_cols = df_work.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
    df_feat = df_work[feat_cols].copy() if feat_cols else df_work.drop(columns=[target_col]).select_dtypes(include=[np.number])

    pre_t1, pre_t2, pre_t3 = st.tabs(["数据概览", "缺失值处理", "特征选择"])

    with pre_t1:
        st.caption("各列统计、分布直方图、相关性热力图。")
        st.write("**基本统计**")
        st.dataframe(df_feat.describe(), use_container_width=True)
        col_hist, col_corr = st.columns(2)
        with col_hist:
            hist_col = st.selectbox("分布直方图 - 选择列", df_feat.columns.tolist(), key="hist_col")
            fig_h, ax_h = plt.subplots(figsize=(5, 3))
            ax_h.hist(df_feat[hist_col].dropna(), bins=30, edgecolor="white", alpha=0.8)
            ax_h.set_xlabel(hist_col)
            ax_h.set_ylabel("频数")
            ax_h.set_title(f"{hist_col} 分布")
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)
        with col_corr:
            if len(df_feat.columns) >= 2:
                corr_mat = df_feat.corr()
                fig_c, ax_c = plt.subplots(figsize=(6, 5))
                im = ax_c.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                ax_c.set_xticks(range(len(corr_mat.columns)))
                ax_c.set_yticks(range(len(corr_mat.columns)))
                ax_c.set_xticklabels(corr_mat.columns, rotation=45, ha="right")
                ax_c.set_yticklabels(corr_mat.columns)
                plt.colorbar(im, ax=ax_c, label="相关系数")
                ax_c.set_title("特征相关性热力图")
                plt.tight_layout()
                st.pyplot(fig_c)
                plt.close(fig_c)
            else:
                st.info("至少需 2 个数值特征才能绘制相关性热力图。")

    with pre_t2:
        missing = df_feat.isnull().sum()
        if missing.sum() == 0:
            st.success("当前数据无缺失值。")
        else:
            st.write("**缺失值统计**")
            st.dataframe(missing[missing > 0].to_frame("缺失数"), use_container_width=True)
            imp_strategy = st.selectbox("填充策略", ["均值", "中位数", "众数", "删除含缺失行"], key="imp_strategy")
            if st.button("应用缺失值处理", key="apply_imp"):
                if imp_strategy == "删除含缺失行":
                    st.session_state[df_key] = df_work.dropna(subset=feat_cols + [target_col]).copy()
                else:
                    strat_map = {"均值": "mean", "中位数": "median", "众数": "most_frequent"}
                    imp = SimpleImputer(strategy=strat_map[imp_strategy])
                    df_work[feat_cols] = imp.fit_transform(df_work[feat_cols])
                    st.session_state[df_key] = df_work.copy()
                st.success("已应用。")
                st.rerun()

    sel_key = f"sel_features_{id(df)}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = None

    with pre_t3:
        sel_method = st.selectbox("特征选择方法", ["不筛选", "方差阈值", "相关性过滤", "基于模型 Top-K"], key="sel_method")
        selected_features = st.session_state.get(sel_key) or list(df_feat.columns)
        if sel_method == "方差阈值":
            var_thresh = st.slider("方差阈值", 0.0, 1.0, 0.01, 0.001, key="var_thresh")
            if st.button("应用方差筛选", key="apply_var"):
                vt = VarianceThreshold(threshold=var_thresh)
                vt.fit(df_feat.fillna(df_feat.mean()))
                selected_features = [df_feat.columns[i] for i in vt.get_support(indices=True)]
                st.session_state[sel_key] = selected_features if selected_features else list(df_feat.columns)
                st.info(f"保留 {len(selected_features)} 个特征")
        elif sel_method == "相关性过滤":
            corr_thresh = st.slider("相关系数阈值（剔除高于此值的冗余特征）", 0.5, 1.0, 0.95, 0.01, key="corr_thresh")
            if st.button("应用相关性筛选", key="apply_corr"):
                corr = df_feat.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
                selected_features = [c for c in df_feat.columns if c not in to_drop]
                st.session_state[sel_key] = selected_features if selected_features else list(df_feat.columns)
                st.info(f"保留 {len(selected_features)} 个特征")
        elif sel_method == "基于模型 Top-K":
            k = st.slider("保留特征数 K", 1, len(df_feat.columns), min(5, len(df_feat.columns)), key="topk")
            if st.button("应用 Top-K 筛选", key="apply_topk"):
                X_tmp = df_feat.fillna(df_feat.mean())
                y_tmp = df_work[target_col]
                if problem_type == "分类" and not np.issubdtype(y_tmp.dtype, np.number):
                    y_tmp = LabelEncoder().fit_transform(y_tmp.astype(str))
                skb = SelectKBest(f_classif if problem_type == "分类" else f_regression, k=min(k, X_tmp.shape[1]))
                skb.fit(X_tmp, y_tmp)
                selected_features = [df_feat.columns[i] for i in skb.get_support(indices=True)]
                st.session_state[sel_key] = selected_features if selected_features else list(df_feat.columns)
                st.info(f"保留 {len(selected_features)} 个特征")
        else:
            st.session_state[sel_key] = None
        feat_cols = st.session_state.get(sel_key) or list(df_feat.columns)
        feat_cols = [f for f in feat_cols if f in df_feat.columns]
    feat_cols = feat_cols if feat_cols else None

    X, y, label_encoder = prepare_X_y(df_work, target_col, problem_type, feat_cols=feat_cols)
    if X.empty or len(X) != len(y):
        st.error("特征或目标异常，请检查目标列。")
        return
    if X.isnull().any().any():
        st.warning("建模数据含缺失值，将用列均值临时填充。建议在「缺失值处理」中先处理。")
        X = X.fillna(X.mean())
    if hasattr(y, "isnull") and y.isnull().any():
        valid_idx = y.dropna().index.intersection(X.index)
        X, y = X.loc[valid_idx], y.loc[valid_idx]

    feature_names = X.columns.tolist()
    st.write("**建模特征列：**", ", ".join(feature_names))
    st.dataframe(df_work[[target_col] + feature_names].head(10), use_container_width=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if problem_type == "分类" and len(np.unique(y)) > 1 else None
    )
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    # 调优参数网格（简化版）
    def _get_param_grid(name, pt):
        if pt == "分类":
            grids = {
                "随机森林": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
                "逻辑回归": {"C": [0.1, 1, 10], "max_iter": [1000]},
                "梯度提升": {"n_estimators": [50, 100], "max_depth": [3, 5]},
                "SVM": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
                "KNN": {"n_neighbors": [3, 5, 10, 20]},
                "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5]},
                "LightGBM": {"n_estimators": [50, 100], "max_depth": [3, 5]},
            }
        else:
            grids = {
                "随机森林": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
                "岭回归": {"alpha": [0.1, 1, 10]},
                "梯度提升": {"n_estimators": [50, 100], "max_depth": [3, 5]},
                "SVM": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
                "KNN": {"n_neighbors": [3, 5, 10, 20]},
                "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5]},
                "LightGBM": {"n_estimators": [50, 100], "max_depth": [3, 5]},
            }
        return grids.get(name, {})

    def _train_single(name, pt, tune=False, cv_folds=None):
        m = get_model(pt, name)
        if tune and _get_param_grid(name, pt):
            from scipy.stats import uniform, randint
            pg = _get_param_grid(name, pt)
            if "随机搜索" in tune_method:
                search = RandomizedSearchCV(m, pg, n_iter=8, cv=tune_cv, random_state=42, n_jobs=-1)
            else:
                search = GridSearchCV(m, pg, cv=tune_cv, n_jobs=-1)
            search.fit(X_train, y_train)
            m = search.best_estimator_
            best_params = search.best_params_
        else:
            best_params = None
            m.fit(X_train, y_train)
        pred = m.predict(X_test)
        cv_scores = None
        if cv_folds:
            scoring = "accuracy" if pt == "分类" else "r2"
            cv_res = cross_validate(m, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
            cv_scores = cv_res["test_score"]
        return m, pred, best_params, cv_scores

    if compare_mode and model_selected:
        models_to_run = model_selected
    else:
        models_to_run = [model_name] if not compare_mode else [model_names[0]]

    # 缓存 key：参数与数据指纹
    cache_parts = (problem_type, tuple(sorted(models_to_run)), test_size, use_scaling, enable_tuning, enable_cv,
                   X.shape[0], X.shape[1], target_col, tuple(sorted(feature_names)))
    cache_key = hashlib.md5(pickle.dumps(cache_parts)).hexdigest()
    cache_store = st.session_state.get("train_cache", {})
    if use_cache and cache_key in cache_store:
        results = cache_store[cache_key]
        model = results[0]["model"]
        y_pred = results[0]["y_pred"]
        model_name = results[0]["name"]
        if compare_mode and len(results) > 1:
            best_idx = np.argmax([accuracy_score(y_test, r["y_pred"]) for r in results]) if problem_type == "分类" else np.argmax([r2_score(y_test, r["y_pred"]) for r in results])
            model, y_pred, model_name = results[best_idx]["model"], results[best_idx]["y_pred"], results[best_idx]["name"]
        st.caption("✅ 使用缓存结果")
    else:
        results = []
        n_models = len(models_to_run)
        progress_bar = st.progress(0, text="准备训练...")
        for i, mn in enumerate(models_to_run):
            progress_bar.progress((i + 1) / n_models, text=f"训练 {mn} ({i+1}/{n_models})...")
            with st.spinner(f"训练 {mn}..."):
                m, pred, best_params, cv_scores = _train_single(
                    mn, problem_type,
                    tune=enable_tuning and _get_param_grid(mn, problem_type),
                    cv_folds=cv_folds if enable_cv else None
                )
            results.append({
                "name": mn,
                "model": m,
                "y_pred": pred,
                "best_params": best_params,
                "cv_scores": cv_scores,
            })
        progress_bar.empty()
        if use_cache:
            cache_store = st.session_state.get("train_cache", {})
            cache_store[cache_key] = results
            st.session_state["train_cache"] = cache_store
    model = results[0]["model"]
    y_pred = results[0]["y_pred"]
    model_name = results[0]["name"]

    st.subheader("模型评估")

    if compare_mode and len(results) > 1:
        # 模型对比表格
        cmp_data = []
        for r in results:
            pred = r["y_pred"]
            row = {"模型": r["name"]}
            if problem_type == "分类":
                row["准确率"] = f"{accuracy_score(y_test, pred):.4f}"
                row["F1 (weighted)"] = f"{f1_score(y_test, pred, average='weighted'):.4f}"
            else:
                row["R²"] = f"{r2_score(y_test, pred):.4f}"
                row["RMSE"] = f"{np.sqrt(mean_squared_error(y_test, pred)):.4f}"
                row["MAE"] = f"{mean_absolute_error(y_test, pred):.4f}"
            if r["cv_scores"] is not None:
                sc = "accuracy" if problem_type == "分类" else "r2"
                row[f"CV {sc} (mean±std)"] = f"{r['cv_scores'].mean():.4f} ± {r['cv_scores'].std():.4f}"
            if r["best_params"]:
                row["最优参数"] = str(r["best_params"])
            cmp_data.append(row)
        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True)
        best_idx = 0
        if problem_type == "分类":
            best_idx = np.argmax([accuracy_score(y_test, r["y_pred"]) for r in results])
        else:
            best_idx = np.argmax([r2_score(y_test, r["y_pred"]) for r in results])
        st.info(f"**最佳模型**：{results[best_idx]['name']}（将用于 SHAP 分析）")
        model = results[best_idx]["model"]
        y_pred = results[best_idx]["y_pred"]
        model_name = results[best_idx]["name"]

    c1, c2, c3 = st.columns(3)
    with c1:
        if problem_type == "分类":
            st.metric("准确率", f"{accuracy_score(y_test, y_pred):.4f}")
        else:
            st.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
    with c2:
        if problem_type == "分类":
            st.metric("F1 (weighted)", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        else:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    with c3:
        if problem_type == "回归":
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")

    if not compare_mode or len(results) == 1:
        if results[0]["best_params"]:
            st.caption("**最优参数**：")
            st.json(results[0]["best_params"])
        if results[0]["cv_scores"] is not None:
            sc = "accuracy" if problem_type == "分类" else "r2"
            st.caption(f"**{cv_folds} 折交叉验证 {sc}**：{results[0]['cv_scores'].mean():.4f} ± {results[0]['cv_scores'].std():.4f}")

    if problem_type == "分类":
        st.text("分类报告")
        st.code(classification_report(y_test, y_pred))
        # 混淆矩阵热力图
        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(np.concatenate([y_test, y_pred]))
        if label_encoder is not None:
            class_labels = [str(label_encoder.inverse_transform([c])[0]) for c in classes]
        else:
            class_labels = [str(c) for c in classes]
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
        ax_cm.set_xticks(range(len(classes)))
        ax_cm.set_yticks(range(len(classes)))
        ax_cm.set_xticklabels(class_labels)
        ax_cm.set_yticklabels(class_labels)
        ax_cm.set_xlabel("预测")
        ax_cm.set_ylabel("真实")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax_cm, label="样本数")
        ax_cm.set_title("混淆矩阵")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # ROC 与 PR 曲线（需 predict_proba）
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            n_classes = len(classes)
            if n_classes == 2:
                y_prob_pos = y_prob[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
                roc_auc = auc(fpr, tpr)
                prec, rec, _ = precision_recall_curve(y_test, y_prob_pos)
                pr_auc = average_precision_score(y_test, y_prob_pos)
                ev_col1, ev_col2 = st.columns(2)
                with ev_col1:
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
                    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
                    ax_roc.set_xlabel("假阳性率")
                    ax_roc.set_ylabel("真阳性率")
                    ax_roc.set_title("ROC 曲线")
                    ax_roc.legend(loc="lower right")
                    ax_roc.set_xlim([0, 1])
                    ax_roc.set_ylim([0, 1])
                    plt.tight_layout()
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)
                with ev_col2:
                    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
                    ax_pr.plot(rec, prec, color="darkgreen", lw=2, label=f"PR (AP={pr_auc:.3f})")
                    ax_pr.set_xlabel("召回率")
                    ax_pr.set_ylabel("精确率")
                    ax_pr.set_title("PR 曲线")
                    ax_pr.legend(loc="upper right")
                    ax_pr.set_xlim([0, 1])
                    ax_pr.set_ylim([0, 1])
                    plt.tight_layout()
                    st.pyplot(fig_pr)
                    plt.close(fig_pr)
            else:
                y_bin = label_binarize(y_test, classes=classes)
                fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
                prec_dict, rec_dict, pr_auc_dict = {}, {}, {}
                for i in range(n_classes):
                    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
                    prec_dict[i], rec_dict[i], _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                    pr_auc_dict[i] = average_precision_score(y_bin[:, i], y_prob[:, i])
                ev_col1, ev_col2 = st.columns(2)
                with ev_col1:
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    for i in range(n_classes):
                        ax_roc.plot(fpr_dict[i], tpr_dict[i], lw=2, label=f"{class_labels[i]} (AUC={roc_auc_dict[i]:.3f})")
                    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
                    ax_roc.set_xlabel("假阳性率")
                    ax_roc.set_ylabel("真阳性率")
                    ax_roc.set_title("ROC 曲线（多分类）")
                    ax_roc.legend(loc="lower right", fontsize=8)
                    ax_roc.set_xlim([0, 1])
                    ax_roc.set_ylim([0, 1])
                    plt.tight_layout()
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)
                with ev_col2:
                    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
                    for i in range(n_classes):
                        ax_pr.plot(rec_dict[i], prec_dict[i], lw=2, label=f"{class_labels[i]} (AP={pr_auc_dict[i]:.3f})")
                    ax_pr.set_xlabel("召回率")
                    ax_pr.set_ylabel("精确率")
                    ax_pr.set_title("PR 曲线（多分类）")
                    ax_pr.legend(loc="upper right", fontsize=8)
                    ax_pr.set_xlim([0, 1])
                    ax_pr.set_ylim([0, 1])
                    plt.tight_layout()
                    st.pyplot(fig_pr)
                    plt.close(fig_pr)
        else:
            st.caption("当前模型不支持 predict_proba，无法绘制 ROC/PR 曲线。")
    else:
        # 回归：残差图、预测 vs 真实散点图
        ev_col1, ev_col2 = st.columns(2)
        residuals = np.array(y_test) - np.array(y_pred)
        with ev_col1:
            fig_res, ax_res = plt.subplots(figsize=(5, 4))
            ax_res.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
            ax_res.axhline(0, color="red", linestyle="--", lw=1)
            ax_res.set_xlabel("预测值")
            ax_res.set_ylabel("残差")
            ax_res.set_title("残差图")
            plt.tight_layout()
            st.pyplot(fig_res)
            plt.close(fig_res)
        with ev_col2:
            fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
            ax_scatter.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y=x")
            ax_scatter.set_xlabel("真实值")
            ax_scatter.set_ylabel("预测值")
            ax_scatter.set_title("预测 vs 真实")
            ax_scatter.legend()
            ax_scatter.set_aspect("equal", adjustable="box")
            plt.tight_layout()
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)

    st.subheader("SHAP 可解释性分析")
    if not SHAP_AVAILABLE:
        st.warning("请安装: pip install shap")
        return

    n_background = min(100, len(X_train))
    n_explain = min(200, len(X_test))
    X_background = X_train.sample(n=n_background, random_state=42)
    X_explain = X_test.head(n_explain).copy()

    shap_cache_key = cache_key + "_shap"
    shap_cache = st.session_state.get("shap_cache", {})
    if use_cache and shap_cache_key in shap_cache:
        explainer, shap_values, X_explain = shap_cache[shap_cache_key]
        st.caption("✅ SHAP 使用缓存")
    else:
        with st.spinner("正在计算 SHAP 值..."):
            explainer, shap_values, X_explain = run_shap_analysis(model, X_background, X_explain, problem_type)
        if use_cache and explainer is not None and shap_values is not None:
            shap_cache[shap_cache_key] = (explainer, shap_values, X_explain)
            st.session_state["shap_cache"] = shap_cache
    if explainer is None or shap_values is None:
        return

    # 多分类：list 或 3D (n_samples, n_features, n_classes)
    sv = np.asarray(shap_values)
    is_multiclass = isinstance(shap_values, list) or (sv.ndim == 3 and sv.shape[-1] > 1)
    n_classes = len(shap_values) if isinstance(shap_values, list) else (sv.shape[-1] if sv.ndim == 3 else 1)
    class_idx = 0
    if is_multiclass and problem_type == "分类":
        class_idx = st.sidebar.selectbox(
            "SHAP 解释类别（多分类）",
            options=list(range(n_classes)),
            format_func=lambda i: f"类别 {i}",
            key="shap_class"
        )
    st.sidebar.markdown("---")
    st.sidebar.subheader("图形保存")
    save_format = st.sidebar.selectbox(
        "保存格式",
        ["PNG", "PDF", "SVG", "EPS"],
        format_func=lambda x: f"{x}（矢量可编辑）" if x in ("PDF", "SVG", "EPS") else x,
        key="save_format"
    )
    st.sidebar.subheader("自定义样式")
    fig_dpi = st.sidebar.slider("PNG DPI", 72, 300, 150, key="fig_dpi")
    fig_fontsize = st.sidebar.slider("字体大小", 8, 16, 10, key="fig_fontsize")
    plt.rcParams["font.size"] = fig_fontsize

    def _gen_zip():
        """生成所有 SHAP 图并打包为 ZIP。"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            ext, _ = FIG_FORMAT_MAP.get(save_format.strip().upper(), FIG_FORMAT_MAP["PNG"])
            plots_done = []
            try:
                shap.summary_plot(shap_arr, X_explain, plot_type="bar", show=False, max_display=min(15, X_explain.shape[1]))
                plt.tight_layout()
                fig = plt.gcf()
                data, fname = save_fig_bytes(fig, save_format, "01_shap_summary_bar", dpi=fig_dpi)
                zf.writestr(fname, data)
                plots_done.append(fname)
                plt.close(fig)
            except Exception:
                pass
            try:
                shap.summary_plot(shap_arr, X_explain, show=False, max_display=min(15, X_explain.shape[1]))
                plt.tight_layout()
                fig = plt.gcf()
                data, fname = save_fig_bytes(fig, save_format, "02_shap_summary_beeswarm", dpi=fig_dpi)
                zf.writestr(fname, data)
                plots_done.append(fname)
                plt.close(fig)
            except Exception:
                pass
            try:
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = ev[class_idx]
                ev = float(ev)
                exp = shap.Explanation(values=shap_arr, base_values=np.full(len(shap_arr), ev), data=X_explain_np, feature_names=feature_names)
                shap.plots.heatmap(exp, max_display=min(10, X_explain.shape[1]), show=False)
                plt.tight_layout()
                fig = plt.gcf()
                data, fname = save_fig_bytes(fig, save_format, "03_shap_heatmap", dpi=fig_dpi)
                zf.writestr(fname, data)
                plots_done.append(fname)
                plt.close(fig)
            except Exception:
                pass
            try:
                shap_importance = np.abs(shap_arr).mean(axis=0)
                shap_importance = shap_importance / (shap_importance.sum() + 1e-9)
                fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
                idx = np.argsort(shap_importance)[::-1][:min(15, len(feature_names))]
                ax_fi.barh([feature_names[i] for i in idx], shap_importance[idx], color="steelblue", alpha=0.8)
                ax_fi.set_xlabel("SHAP 平均绝对贡献（归一化）")
                ax_fi.set_title("SHAP 特征重要性")
                plt.tight_layout()
                data, fname = save_fig_bytes(fig_fi, save_format, "04_shap_importance", dpi=fig_dpi)
                zf.writestr(fname, data)
                plots_done.append(fname)
                plt.close(fig_fi)
            except Exception:
                pass
            if shap_interaction is not None:
                try:
                    shap.summary_plot(shap_interaction, X_explain_np, feature_names=feature_names, max_display=min(7, X_explain.shape[1]), show=False)
                    plt.tight_layout()
                    fig = plt.gcf()
                    data, fname = save_fig_bytes(fig, save_format, "05_shap_interaction_values", dpi=fig_dpi)
                    zf.writestr(fname, data)
                    plots_done.append(fname)
                    plt.close(fig)
                except Exception:
                    pass
        buf.seek(0)
        return buf.getvalue(), len(plots_done)

    zip_key = "shap_zip_bytes"
    if st.sidebar.button("📦 批量导出 SHAP 图为 ZIP", key="batch_export"):
        with st.spinner("正在生成图形..."):
            zip_bytes, _ = _gen_zip()
            st.session_state[zip_key] = zip_bytes
    if zip_key in st.session_state:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.sidebar.download_button("下载 ZIP", data=st.session_state[zip_key], file_name=f"shap_plots_{ts}.zip", mime="application/zip", key="dl_zip")
    shap_arr = _shap_to_2d(shap_values, class_idx)
    X_explain_np = np.asarray(X_explain) if not isinstance(X_explain, np.ndarray) else X_explain

    # 树模型 SHAP 交互值（仅 TreeExplainer 支持）
    shap_interaction = None
    if hasattr(explainer, "shap_interaction_values") and _is_tree_model(model):
        try:
            si = explainer.shap_interaction_values(X_explain)
            if isinstance(si, list):
                shap_interaction = si[class_idx] if class_idx < len(si) else si[0]
            else:
                shap_interaction = si
        except Exception:
            shap_interaction = None

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "SHAP 摘要图", "特征依赖图", "交互图", "SHAP 交互值", "热力图", "决策图", "小提琴图", "瀑布图", "力图", "特征重要性对比"
    ])

    with tab1:
        st.caption("各特征对预测的贡献。")
        plot_type = st.radio("摘要图类型", ["条形图 (bar)", "蜂群图 (beeswarm)"], horizontal=True)
        try:
            if plot_type.startswith("条形"):
                shap.summary_plot(shap_arr, X_explain, plot_type="bar", show=False, max_display=min(15, X_explain.shape[1]))
            else:
                shap.summary_plot(shap_arr, X_explain, show=False, max_display=min(15, X_explain.shape[1]))
            plt.tight_layout()
            fig = plt.gcf()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_summary", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab1")
            plt.close(fig)
        except Exception as e:
            st.error(f"摘要图绘制失败: {e}")

    with tab2:
        dep_feature = st.selectbox("选择特征", feature_names, key="dep_feat")
        dep_idx = feature_names.index(dep_feature)
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                dep_idx,
                shap_arr,
                X_explain_np,
                feature_names=feature_names,
                ax=ax,
                show=False,
                interaction_index=None,
            )
            plt.tight_layout()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_dependence", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab2")
            plt.close(fig)
        except Exception as e:
            st.error(f"依赖图绘制失败: {e}")

    with tab3:
        st.caption("交互图：展示特征与另一特征的交互效应，颜色表示交互特征取值。")
        inter_feature = st.selectbox("选择主特征", feature_names, key="inter_feat")
        inter_idx = feature_names.index(inter_feature)
        inter_color = st.selectbox("交互着色特征", ["自动选择", "无"] + feature_names, key="inter_color")
        interaction_index = None if inter_color == "无" else ("auto" if inter_color == "自动选择" else inter_color)
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                inter_idx,
                shap_arr,
                X_explain_np,
                feature_names=feature_names,
                ax=ax,
                show=False,
                interaction_index=interaction_index,
            )
            plt.tight_layout()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_interaction", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab3")
            plt.close(fig)
        except Exception as e:
            st.error(f"交互图绘制失败: {e}")

    with tab4:
        st.caption("SHAP 交互值：树模型（如随机森林）支持，展示特征间交互效应。")
        if shap_interaction is not None:
            try:
                shap.summary_plot(
                    shap_interaction,
                    X_explain_np,
                    feature_names=feature_names,
                    max_display=min(7, X_explain.shape[1]),
                    show=False,
                )
                plt.tight_layout()
                fig = plt.gcf()
                fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_interaction_values", dpi=fig_dpi)
                _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
                st.pyplot(fig)
                st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab4")
                plt.close(fig)
            except Exception as e:
                st.error(f"SHAP 交互值图绘制失败: {e}")
        else:
            st.info("当前模型不支持 SHAP 交互值（需树模型如随机森林）。")

    with tab5:
        st.caption("热力图：按样本展示各特征的 SHAP 值，样本按聚类排序。")
        try:
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = np.array(ev)[class_idx]
            ev = float(ev)
            exp = shap.Explanation(
                values=shap_arr,
                base_values=np.full(len(shap_arr), ev),
                data=X_explain_np,
                feature_names=feature_names
            )
            shap.plots.heatmap(exp, max_display=min(10, X_explain.shape[1]), show=False)
            plt.tight_layout()
            fig = plt.gcf()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_heatmap", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab5")
            plt.close(fig)
        except Exception as e:
            st.error(f"热力图绘制失败: {e}")

    with tab6:
        st.caption("决策图：展示各样本的累积 SHAP 值变化，每条线代表一个样本。")
        n_decision = st.slider("决策图样本数（过多会难以阅读）", 5, 50, 20, key="n_decision")
        X_decision = X_explain.head(n_decision)
        shap_decision = shap_arr[:n_decision]
        try:
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = ev[class_idx]
            ev = float(ev)
            shap.decision_plot(
                ev, shap_decision, X_decision.values,
                feature_names=feature_names,
                feature_order="importance",
                show=False,
                ignore_warnings=True
            )
            plt.tight_layout()
            fig = plt.gcf()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_decision", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab6")
            plt.close(fig)
        except Exception as e:
            st.error(f"决策图绘制失败: {e}")

    with tab7:
        st.caption("小提琴图：各特征 SHAP 值的分布。")
        violin_type = st.radio("类型", ["violin", "layered_violin"], horizontal=True, key="violin_type")
        try:
            shap.plots.violin(
                shap_arr, X_explain,
                feature_names=feature_names,
                plot_type=violin_type,
                show=False,
                max_display=min(15, X_explain.shape[1])
            )
            plt.tight_layout()
            fig = plt.gcf()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_violin", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab7")
            plt.close(fig)
        except Exception as e:
            st.error(f"小提琴图绘制失败: {e}")

    with tab8:
        st.caption("瀑布图：单样本各特征对预测的贡献分解。")
        sample_idx = st.slider("测试集样本索引", 0, min(50, len(X_explain) - 1), 0, key="waterfall_idx")
        row = X_explain.iloc[sample_idx : sample_idx + 1]
        row_1d = row.iloc[0]
        try:
            single_shap = explainer.shap_values(row, check_additivity=False) if _is_tree_model(model) else explainer.shap_values(row)
            single_shap_2d = _shap_to_2d(single_shap, class_idx)
            single_shap_1d = np.squeeze(single_shap_2d)
            if single_shap_1d.ndim > 1:
                single_shap_1d = single_shap_1d[0]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = ev[class_idx]
            ev = float(ev)
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values=single_shap_1d,
                    base_values=ev,
                    data=row_1d.values,
                    feature_names=feature_names
                ),
                show=False
            )
            plt.tight_layout()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_waterfall", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab8")
            plt.close(fig)
        except Exception as e:
            st.error(f"瀑布图绘制失败: {e}")

    with tab9:
        st.caption("力图：单样本各特征对预测的推动/拉低作用。")
        sample_idx = st.slider("测试集样本索引", 0, min(50, len(X_explain) - 1), 0, key="force_idx")
        row = X_explain.iloc[sample_idx : sample_idx + 1]
        row_1d = row.iloc[0]
        try:
            single_shap = explainer.shap_values(row, check_additivity=False) if _is_tree_model(model) else explainer.shap_values(row)
            single_shap_2d = _shap_to_2d(single_shap, class_idx)
            single_shap_1d = np.squeeze(single_shap_2d)
            if single_shap_1d.ndim > 1:
                single_shap_1d = single_shap_1d[0]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = ev[class_idx]
            ev = float(ev)
            features_display = np.array([f"{float(v):.2f}" for v in row_1d.values])
            shap.plots.force(ev, single_shap_1d, features_display, feature_names=feature_names, matplotlib=True, show=False)
            plt.tight_layout()
            fig = plt.gcf()
            fig_bytes, fname = save_fig_bytes(fig, save_format, "shap_force", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab9")
            plt.close(fig)
        except Exception as e:
            st.error(f"力图绘制失败: {e}")

    with tab10:
        st.caption("对比 SHAP 平均绝对贡献与模型自带的 feature_importances_（树模型）。")
        shap_importance = np.abs(shap_arr).mean(axis=0)
        shap_importance = shap_importance / (shap_importance.sum() + 1e-9)
        has_fi = hasattr(model, "feature_importances_")
        if has_fi:
            fi = model.feature_importances_
            fi = fi / (fi.sum() + 1e-9)
            fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
            x = np.arange(len(feature_names))
            w = 0.35
            ax_fi.bar(x - w / 2, shap_importance, w, label="SHAP (mean |SHAP|)", color="steelblue", alpha=0.8)
            ax_fi.bar(x + w / 2, fi, w, label="feature_importances_", color="coral", alpha=0.8)
            ax_fi.set_xticks(x)
            ax_fi.set_xticklabels(feature_names, rotation=45, ha="right")
            ax_fi.set_ylabel("归一化重要性")
            ax_fi.set_title("SHAP vs feature_importances_ 对比")
            ax_fi.legend()
            plt.tight_layout()
            fig_bytes, fname = save_fig_bytes(fig_fi, save_format, "feature_importance_compare", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig_fi)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab10")
            plt.close(fig_fi)
        else:
            st.info("当前模型（如逻辑回归、岭回归）无 feature_importances_，仅展示 SHAP 重要性。")
            fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
            idx = np.argsort(shap_importance)[::-1][:min(15, len(feature_names))]
            ax_fi.barh([feature_names[i] for i in idx], shap_importance[idx], color="steelblue", alpha=0.8)
            ax_fi.set_xlabel("SHAP 平均绝对贡献（归一化）")
            ax_fi.set_title("SHAP 特征重要性")
            plt.tight_layout()
            fig_bytes, fname = save_fig_bytes(fig_fi, save_format, "shap_importance", dpi=fig_dpi)
            _, mime = FIG_FORMAT_MAP.get(save_format, FIG_FORMAT_MAP["PNG"])
            st.pyplot(fig_fi)
            st.download_button("下载图形", data=fig_bytes, file_name=fname, mime=mime, key="dl_tab10")
            plt.close(fig_fi)


if __name__ == "__main__":
    main()
