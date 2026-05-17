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
