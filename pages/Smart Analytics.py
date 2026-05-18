import numpy as npy
import pandas as pds
import streamlit as slt


slt.set_page_config(page_title="AI Analytics", layout="wide")


def procssng(daf):
    processed_daf = daf.copy()
    processed_daf.columns = [str(column).strip() for column in processed_daf.columns]

    missing_before = processed_daf.isna().sum()

    unnamed_columns = [
        column
        for column in processed_daf.columns
        if str(column).lower().startswith("unnamed")
    ]
    empty_unnamed_columns = [
        column
        for column in unnamed_columns
        if processed_daf[column].isna().all()
        or processed_daf[column].fillna("").astype(str).str.strip().eq("").all()
    ]

    if empty_unnamed_columns:
        processed_daf = processed_daf.drop(columns=empty_unnamed_columns)

    processed_daf = processed_daf.replace([npy.inf, -npy.inf], npy.nan)

    filled_columns = []

    for column in processed_daf.columns:
        if not processed_daf[column].isna().any():
            continue

        if pds.api.types.is_numeric_dtype(processed_daf[column]):
            median_value = processed_daf[column].median()
            fill_value = 0 if pds.isna(median_value) else median_value
            processed_daf[column] = processed_daf[column].fillna(fill_value)
        else:
            processed_daf[column] = processed_daf[column].fillna("")

        filled_columns.append(column)

    processing_report = {
        "rows": processed_daf.shape[0],
        "columns_before": daf.shape[1],
        "columns_after": processed_daf.shape[1],
        "missing_before": missing_before,
        "missing_after": processed_daf.isna().sum(),
        "dropped_columns": empty_unnamed_columns,
        "filled_columns": filled_columns,
    }

    return processed_daf, processing_report


def load_csv(uploaded_file):
    for separator in [",", ";", "\t"]:
        uploaded_file.seek(0)
        daf = pds.read_csv(uploaded_file, sep=separator)

        if daf.shape[1] > 1:
            return daf

    uploaded_file.seek(0)
    return pds.read_csv(uploaded_file)


def show_processed_dataset(processed_daf, processing_report):
    slt.subheader("Обработанный датасет")

    metric_col_1, metric_col_2, metric_col_3 = slt.columns(3)

    with metric_col_1:
        slt.metric("Строк", processing_report["rows"])

    with metric_col_2:
        slt.metric(
            "Колонок",
            processing_report["columns_after"],
            delta=processing_report["columns_after"] - processing_report["columns_before"],
        )

    with metric_col_3:
        slt.metric("Пропусков после обработки", int(processing_report["missing_after"].sum()))

    if processing_report["dropped_columns"]:
        slt.info(
            "Удалены пустые колонки: "
            + ", ".join(processing_report["dropped_columns"])
        )

    if processing_report["filled_columns"]:
        slt.info(
            "Заполнены пропуски: "
            + ", ".join(processing_report["filled_columns"])
        )

    slt.dataframe(
        processed_daf,
        use_container_width=True,
        hide_index=True,
    )


slt.header("Модуль автоматизированной аналитики")

uploaded_file = slt.file_uploader(
    "Загрузите CSV-файл",
    type=["csv"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    raw_daf = load_csv(uploaded_file)
    processed_daf, processing_report = procssng(raw_daf)
    show_processed_dataset(processed_daf, processing_report)
else:
    slt.info("Загрузите CSV-файл.")

slt.text_area(
    "Запрос к модулю",
    placeholder="",
    height=120,
    disabled=True,
)


class OnlineMCamp:
    def __init__(self, daf):
        self.daf = daf.copy()
        self.required_roi_columns = ["cost", "post_click_conversions"]

    def check_columns(self, columns):
        missing_columns = [
            column
            for column in columns
            if column not in self.daf.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing columns for online marketing analysis: "
                + ", ".join(missing_columns)
            )

    def add_roi(self):
        self.check_columns(self.required_roi_columns)

        self.daf["ROI"] = npy.where(
            self.daf["post_click_conversions"] > 0,
            self.daf["cost"] / self.daf["post_click_conversions"],
            0,
        )

        return self.daf[["ROI", "cost", "post_click_conversions"]].copy()

    def get_dataset_with_roi(self):
        if "ROI" not in self.daf.columns:
            self.add_roi()

        return self.daf.copy()

    def best_investments_by_roi(self):
        daf_with_roi = self.get_dataset_with_roi()

        return daf_with_roi.sort_values(
            by="ROI",
            ascending=False,
        ).copy()

    def campaign_cost_avg_roi(self):
        self.check_columns(["campaign_number", "cost"])
        daf_with_roi = self.get_dataset_with_roi()

        return (
            daf_with_roi
            .groupby("campaign_number", as_index=False)
            .agg(
                total_cost=("cost", "sum"),
                avg_roi=("ROI", "mean"),
                conversions=("post_click_conversions", "sum"),
            )
            .sort_values(by="avg_roi", ascending=False)
        )

    def monthly_campaign_cost_roi(self):
        self.check_columns(["month", "campaign_number", "cost"])
        daf_with_roi = self.get_dataset_with_roi()

        return (
            daf_with_roi
            .groupby(["month", "campaign_number"], as_index=False)
            .agg(
                total_cost=("cost", "sum"),
                avg_roi=("ROI", "mean"),
                conversions=("post_click_conversions", "sum"),
            )
            .sort_values(by=["month", "campaign_number"])
        )

    def successful_campaigns_by_roi(self, roi_threshold=1):
        self.check_columns(["campaign_number"])
        daf_with_roi = self.get_dataset_with_roi()
        successful_campaigns = daf_with_roi[daf_with_roi["ROI"] > roi_threshold]

        return (
            successful_campaigns
            .groupby("campaign_number", as_index=False)
            .agg(
                avg_roi=("ROI", "mean"),
                total_cost=("cost", "sum"),
                conversions=("post_click_conversions", "sum"),
            )
            .sort_values(by="avg_roi", ascending=False)
        )

    def placements_by_roi(self):
        self.check_columns(["placement"])
        daf_with_roi = self.get_dataset_with_roi()

        return (
            daf_with_roi
            .groupby("placement", as_index=False)
            .agg(
                avg_roi=("ROI", "mean"),
                total_cost=("cost", "sum"),
                conversions=("post_click_conversions", "sum"),
            )
            .sort_values(by="avg_roi", ascending=False)
        )

    def banners_by_roi(self):
        self.check_columns(["banner"])
        daf_with_roi = self.get_dataset_with_roi()

        return (
            daf_with_roi
            .groupby("banner", as_index=False)
            .agg(
                avg_roi=("ROI", "mean"),
                total_cost=("cost", "sum"),
                conversions=("post_click_conversions", "sum"),
            )
            .sort_values(by="avg_roi", ascending=False)
        )

    def banner_placement_by_roi(self):
        self.check_columns(["banner", "placement"])
        daf_with_roi = self.get_dataset_with_roi()

        return (
            daf_with_roi
            .groupby(["banner", "placement"], as_index=False)
            .agg(
                avg_roi=("ROI", "mean"),
                total_cost=("cost", "sum"),
                conversions=("post_click_conversions", "sum"),
            )
            .sort_values(by="avg_roi", ascending=False)
        )

    def prepare_roi_features(self):
        daf_with_roi = self.get_dataset_with_roi()
        drop_columns = [
            column
            for column in ["ROI", "date"]
            if column in daf_with_roi.columns
        ]

        x = daf_with_roi.drop(columns=drop_columns).copy()
        x = x.replace([npy.inf, -npy.inf], npy.nan)

        for column in x.columns:
            if pds.api.types.is_numeric_dtype(x[column]):
                median_value = x[column].median()
                fill_value = 0 if pds.isna(median_value) else median_value
                x[column] = x[column].fillna(fill_value)
            else:
                x[column] = x[column].fillna("")

        x = pds.get_dummies(x)
        y = daf_with_roi["ROI"]

        return x, y

    def roi_factor_importance(self, n_estimators=100, top_n=10):
        x, y = self.prepare_roi_features()

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
        )
        rf.fit(x, y)

        importances = pds.Series(
            rf.feature_importances_,
            index=x.columns,
            name="importance",
        )

        return (
            importances
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    def roi_pca(self, n_components=2):
        x, y = self.prepare_roi_features()

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_scaled)

        pca_columns = [
            f"PC{component_number}"
            for component_number in range(1, n_components + 1)
        ]
        pca_daf = pds.DataFrame(
            x_pca,
            columns=pca_columns,
            index=x.index,
        )
        pca_daf["ROI"] = y.values

        explained_variance = pds.DataFrame({
            "component": pca_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        })

        return pca_daf, explained_variance

    def roi_analysis_pack(self):
        return {
            "roi_table": self.add_roi(),
            "best_investments": self.best_investments_by_roi(),
            "campaign_cost_avg_roi": self.campaign_cost_avg_roi(),
            "monthly_campaign_cost_roi": self.monthly_campaign_cost_roi(),
            "successful_campaigns": self.successful_campaigns_by_roi(),
            "placements_by_roi": self.placements_by_roi(),
            "banners_by_roi": self.banners_by_roi(),
            "banner_placement_by_roi": self.banner_placement_by_roi(),
            "roi_factor_importance": self.roi_factor_importance(),
            "roi_pca": self.roi_pca(),
        }
