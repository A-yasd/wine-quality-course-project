"""Build the course-report DOCX from the generated numeric results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def set_run_font(run, name: str = "宋体", size: float | None = None, bold: bool | None = None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_cell_text(cell, text: str, bold: bool = False, size: float = 8.0, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    set_run_font(run, "宋体", size, bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def shade_cell(cell, fill: str):
    cell_properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell_properties.append(shading)


def set_cell_margins(table, top=60, start=70, bottom=60, end=70):
    table_properties = table._tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for side, value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    table_properties.append(margins)


def add_caption(document: Document, text: str):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(1)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(text)
    set_run_font(run, "宋体", 8.5)
    run.font.color.rgb = RGBColor(70, 70, 70)


def add_body(document: Document, text: str, first_indent: bool = True):
    paragraph = document.add_paragraph(style="Report Body")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Pt(21)
    if not first_indent:
        paragraph.paragraph_format.first_line_indent = Pt(0)
    run = paragraph.add_run(text)
    set_run_font(run, "宋体", 10.5)
    return paragraph


def add_heading(document: Document, text: str, level: int = 1):
    paragraph = document.add_paragraph(style=f"Report Heading {level}")
    run = paragraph.add_run(text)
    set_run_font(run, "黑体", 13.5 if level == 1 else 11.5, True)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(8 if level == 1 else 4)
    paragraph.paragraph_format.space_after = Pt(3)
    return paragraph


def add_equation(document: Document, text: str):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(text)
    set_run_font(run, "Cambria Math", 10)


def add_page_number(section):
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.append(field_begin)
    run._r.append(instruction)
    run._r.append(field_end)
    set_run_font(run, "宋体", 9)


def add_highlighted_placeholder(paragraph, text: str):
    run = paragraph.add_run(text)
    set_run_font(run, "宋体", 9.5)
    highlight = OxmlElement("w:highlight")
    highlight.set(qn("w:val"), "yellow")
    run._r.get_or_add_rPr().append(highlight)


def read_results():
    with (RESULTS_DIR / "summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        summary = {row["model"]: row for row in csv.DictReader(handle)}
    with (RESULTS_DIR / "metrics_by_run.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        run_rows = list(csv.DictReader(handle))
    with (RESULTS_DIR / "feature_importance.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        importance_rows = list(csv.DictReader(handle))
    dataset_summary = json.loads(
        (RESULTS_DIR / "dataset_summary.json").read_text(encoding="utf-8")
    )
    return summary, run_rows, importance_rows, dataset_summary


def paired_comparison(
    run_rows: list[dict[str, str]], base_model: str, comparison_model: str
) -> tuple[float, float]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in run_rows:
        grouped[(row["seed"], row["fold"])][row["model"]] = float(row["macro_f1"])
    keys = sorted(grouped)
    base = np.asarray([grouped[key][base_model] for key in keys])
    comparison = np.asarray([grouped[key][comparison_model] for key in keys])
    test = stats.ttest_rel(base, comparison)
    return float(np.mean(base - comparison)), float(test.pvalue)


def feature_summary(importance_rows: list[dict[str, str]], model_name: str):
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in importance_rows:
        if row["model"] == model_name:
            grouped[row["feature"]].append(float(row["pr_auc_drop"]))
    summary = []
    for feature, values in grouped.items():
        array = np.asarray(values)
        summary.append((feature, float(array.mean()), float(array.std(ddof=1))))
    return sorted(summary, key=lambda item: item[1], reverse=True)


def cell_value(summary, model_name: str, metric: str) -> str:
    row = summary[model_name]
    return f"{float(row[f'{metric}_mean']):.3f} ± {float(row[f'{metric}_std']):.3f}"


def make_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    widths_cm: list[float] | None = None,
):
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = True
    set_cell_margins(table)
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, size=8.0)
        shade_cell(table.rows[0].cells[index], "D9EAF7")
    for row_values in rows:
        row_cells = table.add_row().cells
        for index, value in enumerate(row_values):
            set_cell_text(row_cells[index], value, size=8.0)
    if widths_cm is not None:
        for row in table.rows:
            for index, width in enumerate(widths_cm):
                row.cells[index].width = Cm(width)
    return table


def build_report(
    student_id: str = "QSZ20262031",
    student_name: str = "黄海星",
    github_url: str = "",
) -> Path:
    summary, run_rows, importance_rows, dataset_summary = read_results()
    best_model = "mlp_ce"
    delta_logistic, p_logistic = paired_comparison(run_rows, best_model, "logistic")
    delta_sqrt, p_sqrt = paired_comparison(run_rows, best_model, "mlp_cw_sqrt")
    top_features = feature_summary(importance_rows, best_model)
    output_path = PROJECT_ROOT.parent / "类别不平衡优质红酒筛选_课程设计.docx"

    document = Document()
    section = document.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(1.65)
    section.bottom_margin = Cm(1.55)
    section.left_margin = Cm(1.9)
    section.right_margin = Cm(1.9)
    section.header_distance = Cm(0.7)
    section.footer_distance = Cm(0.7)
    add_page_number(section)

    normal = document.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    normal.paragraph_format.space_after = Pt(2.5)

    for level in (1, 2):
        style = document.styles.add_style(f"Report Heading {level}", 1)
        style.base_style = normal
    body_style = document.styles.add_style("Report Body", 1)
    body_style.base_style = normal

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("深度学习课程设计 · 2026 秋季")
    set_run_font(run, "宋体", 8.5)
    run.font.color.rgb = RGBColor(110, 110, 110)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run("类别不平衡场景下基于多层感知机的优质红酒筛选")
    set_run_font(run, "黑体", 17, True)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(5)
    run = subtitle.add_run("——类别权重与阈值校准的对比研究")
    set_run_font(run, "楷体", 11, False)

    info = document.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.paragraph_format.space_after = Pt(6)
    run = info.add_run(f"学号：{student_id}    姓名：{student_name}    GitHub 代码链接：")
    set_run_font(run, "宋体", 9.5)
    if github_url:
        run = info.add_run(github_url)
        set_run_font(run, "宋体", 9.0)
    else:
        add_highlighted_placeholder(info, "仓库创建后填写")

    abstract_title = document.add_paragraph()
    abstract_title.paragraph_format.space_before = Pt(2)
    abstract_title.paragraph_format.space_after = Pt(1)
    run = abstract_title.add_run("摘  要")
    set_run_font(run, "黑体", 10.5, True)
    abstract = document.add_paragraph()
    abstract.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abstract.paragraph_format.first_line_indent = Pt(21)
    abstract.paragraph_format.space_after = Pt(2)
    run = abstract.add_run(
        "红酒质量评级通常依赖人工品评和实验室检测，而高质量样本只占少数，普通准确率容易被多数类掩盖。"
        "本文以 UCI 红酒理化指标数据为对象，将“质量评分≥7”定义为优质类，比较加权逻辑回归与三种正类权重设置下的多层感知机。"
        "实验采用 5 折分层外验证并重复 3 个随机种子，在训练折内部完成标准化、早停和决策阈值选择。"
        "结果显示，无权重 MLP 获得最高 Macro-F1（0.727±0.045）和 PR-AUC（0.538±0.086），相较逻辑回归的配对差值为 0.030（p=0.029）；"
        "继续增大正类权重会提高固定阈值下的少数类召回，却没有稳定提升综合指标。"
        "置换重要性分析表明，酒精度和硫酸盐是模型最主要的判别依据。"
    )
    set_run_font(run, "宋体", 9.3)
    keywords = document.add_paragraph()
    keywords.paragraph_format.space_after = Pt(4)
    run = keywords.add_run("关键词：类别不平衡；多层感知机；代价敏感学习；PR-AUC；红酒质量")
    set_run_font(run, "宋体", 9.0)

    add_heading(document, "1  研究问题")
    add_body(
        document,
        "红酒质量与酒精度、酸度、含糖量、二氧化硫等理化指标相关。若能在不依赖完整人工品评的情况下先筛出潜在高质量样本，"
        "就可以把后续检测资源集中在更可能产生高评分的批次上。该问题同时具有明显的类别不平衡特征：多数红酒处于中等质量，"
        "真正达到 7 分及以上的样本较少。若只追求 accuracy，模型可以通过把几乎所有样本判为“非优质”而得到虚高结果，"
        "但对实际筛选没有帮助。"
    )
    add_body(
        document,
        "本文围绕两个问题展开：一是对于仅 11 维、约 1600 个样本的小规模任务，小型神经网络能否比线性模型更好地刻画理化指标间的非线性交互；"
        "二是代价敏感学习是否能在校准决策阈值后稳定改善稀有类识别。第二个问题刻意设置为可被证伪：若权重只改变概率尺度而不改善排序，"
        "那么经过阈值校准后收益可能消失。这一设计强调对比和证据，而不是预设某种方法必然更优。"
    )

    add_heading(document, "2  数据与实验环境")
    add_body(
        document,
        "数据来自 UCI Machine Learning Repository 的 Wine Quality 数据集，使用其中的红葡萄酒子集。原始数据含 1599 个样本、11 个连续理化特征，"
        "质量标签为整数 3–8。为避免红、白葡萄酒两个分布被混合后产生额外域差异，本文只使用红酒子集；原始标签≥7 的 217 个样本定义为优质类，"
        "其余 1382 个为非优质类，正类率为 13.57%，负正样本比约为 6.37:1。"
    )
    table_rows = [
        [
            f"{dataset_summary['n_samples']}（217 / 1382）",
            "13.57%",
            "6.37 : 1",
            "11 个连续特征；标签 3–8",
            "Python 3.11；NumPy；Matplotlib",
        ]
    ]
    make_table(
        document,
        ["样本数（优 / 非优）", "正类率", "不平衡比", "任务结构", "运行环境"],
        table_rows,
        [3.0, 2.0, 2.1, 4.0, 4.6],
    )
    add_caption(document, "表 1  数据集与实验环境")
    document.add_picture(
        str(FIGURES_DIR / "fig_dataset.png"), width=Inches(6.55)
    )
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "图 1  质量评分分布与特征—标签的线性关联")

    add_heading(document, "3  方法与技术")
    add_body(
        document,
        "首先采用带权重的一元逻辑回归作为可解释基线。其决策边界是理化特征的线性组合，便于判断神经网络带来的增益是否来自非线性建模。"
        "随后使用 11–32–16–1 的三层感知机，以 ReLU 激活和 Sigmoid 输出表示优质概率。输入维度低、样本量小，因此没有使用更深的网络，"
        "而是通过 L2 正则、早停和固定随机种子控制过拟合与复现性。"
    )
    add_equation(
        document,
        "p(x) = Sigmoid(W3*ReLU(W2*ReLU(W1*x + b1) + b2) + b3)",
    )
    add_body(
        document,
        "为研究代价敏感学习，令正类样本权重为 w，负类权重为 1。MLP 使用如下加权二元交叉熵；逻辑回归使用相同损失，但将隐藏层替换为线性映射。"
    )
    add_equation(
        document,
        "L = -(1/N) * sum_i[w*y_i*ln(p_i) + (1-y_i)*ln(1-p_i)] + (lambda/2)*sum(W^2)",
    )
    add_body(
        document,
        "设置三个消融条件：w=1（无权重）、w=√(Nneg/Npos)（温和加权）和 w=Nneg/Npos（完整类比）。"
        "训练采用 Adam，学习率 0.003、批大小 128，最多 260 轮；以内部验证集 PR-AUC 触发早停。"
        "最终分类阈值不是固定为 0.5，而是在内部验证集上选择使 Macro-F1 最大的 τ；测试折只用于一次最终评估。"
    )
    add_equation(
        document,
        "tau* = argmax_tau MacroF1(y_val, 1[p_val >= tau])",
    )

    add_heading(document, "4  设计与实现")
    add_body(
        document,
        "实验采用两层嵌套的验证思想：外层 5 折分层交叉验证，每一折的训练集再分层划分 80%/20% 作为内部训练与验证；外层重复 3 个随机种子，共得到 15 次独立评估。"
        "标准化均值与标准差、网络参数、早停轮次及分类阈值全部只由训练折和内部验证折确定。这样可避免测试折分布泄漏到预处理或阈值选择。"
        "所有模型均在 CPU 上以 NumPy 实现，单次完整实验约 30 秒。"
    )
    method_rows = [
        ["数据划分", "外层 5 折 × 3 种子；内层 80%/20% 分层验证"],
        ["MLP 结构", "11–32–16–1；ReLU；Sigmoid；L2=1e-4"],
        ["优化策略", "Adam；学习率 0.003；batch=128；早停 patience=35"],
        ["输出指标", "Macro-F1、PR-AUC、ROC-AUC、优质召回、非优质特异度"],
    ]
    make_table(document, ["环节", "具体设置"], method_rows, [3.0, 12.2])
    add_caption(document, "表 2  实验实现与验证设置")
    add_body(
        document,
        "选择 PR-AUC 作为主要排序指标，是因为在 13.57% 正类率下，ROC-AUC 可能对大量真负例过于乐观；Macro-F1 对两个类别等权，"
        "可约束模型不能通过牺牲多数类换取少数类召回。为解释模型，对外层测试折进行 4 次逐特征置换，记录 PR-AUC 的平均下降量。"
        "该值表示“打乱某特征与标签对应关系后性能损失多少”，是模型依赖度而非因果效应。"
    )

    add_heading(document, "5  实验与结果")
    metric_rows = [
        [
            "加权逻辑回归",
            cell_value(summary, "logistic", "macro_f1"),
            cell_value(summary, "logistic", "pr_auc"),
            cell_value(summary, "logistic", "roc_auc"),
            cell_value(summary, "logistic", "recall"),
            cell_value(summary, "logistic", "specificity"),
            f"{float(summary['logistic']['fixed_recall_mean']):.3f}",
        ],
        [
            "MLP（w=1）",
            cell_value(summary, "mlp_ce", "macro_f1"),
            cell_value(summary, "mlp_ce", "pr_auc"),
            cell_value(summary, "mlp_ce", "roc_auc"),
            cell_value(summary, "mlp_ce", "recall"),
            cell_value(summary, "mlp_ce", "specificity"),
            f"{float(summary['mlp_ce']['fixed_recall_mean']):.3f}",
        ],
        [
            "MLP（w=√比）",
            cell_value(summary, "mlp_cw_sqrt", "macro_f1"),
            cell_value(summary, "mlp_cw_sqrt", "pr_auc"),
            cell_value(summary, "mlp_cw_sqrt", "roc_auc"),
            cell_value(summary, "mlp_cw_sqrt", "recall"),
            cell_value(summary, "mlp_cw_sqrt", "specificity"),
            f"{float(summary['mlp_cw_sqrt']['fixed_recall_mean']):.3f}",
        ],
        [
            "MLP（w=完整比）",
            cell_value(summary, "mlp_cw", "macro_f1"),
            cell_value(summary, "mlp_cw", "pr_auc"),
            cell_value(summary, "mlp_cw", "roc_auc"),
            cell_value(summary, "mlp_cw", "recall"),
            cell_value(summary, "mlp_cw", "specificity"),
            f"{float(summary['mlp_cw']['fixed_recall_mean']):.3f}",
        ],
    ]
    make_table(
        document,
        [
            "模型",
            "Macro-F1",
            "PR-AUC",
            "ROC-AUC",
            "优质召回",
            "非优质特异度",
            "τ=0.5 召回",
        ],
        metric_rows,
        [3.3, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9],
    )
    add_caption(document, "表 3  15 次外层评估的均值 ± 标准差；前六列为各自校准阈值下的结果")
    add_body(
        document,
        "无权重 MLP 的 Macro-F1 为 0.727±0.045，PR-AUC 为 0.538±0.086，两项均高于逻辑回归；按同种子同折配对比较，"
        f"Macro-F1 平均提升 {delta_logistic:.3f}，配对 t 检验 p={p_logistic:.3f}。"
        "逻辑回归的 ROC-AUC 反而最高（0.870），但其 PR-AUC 和 Macro-F1 较低，说明它在整体排序上有效，却未能在高精确率区间更好地区分稀有优质样本。"
        "这体现了不平衡二分类中仅报告 ROC-AUC 的局限。"
    )
    document.add_picture(
        str(FIGURES_DIR / "fig_model_comparison.png"), width=Inches(6.25)
    )
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "图 2  四种模型的 Macro-F1 与 PR-AUC（误差线为标准差）")

    add_heading(document, "6  分析与讨论")
    add_body(
        document,
        "权重消融给出了比“加权通常能改善不平衡分类”更细的结论。在固定阈值 0.5 下，完整类比权重把优质召回从 0.427 提高到 0.754，"
        "但非优质特异度从 0.951 降至 0.809，固定阈值 Macro-F1 反而下降。经过逐折阈值校准后，全部加权 MLP 都未超过无权重 MLP："
        f"温和加权与无权重模型的 Macro-F1 差值为 {delta_sqrt:.3f}，配对 p={p_sqrt:.3f}。"
        "原因是正类权重主要重新缩放输出概率，而阈值 τ 本就可以在验证集上移动决策边界；当排序质量未同步提升时，二者作用高度重叠。"
    )
    document.add_picture(
        str(FIGURES_DIR / "fig_analysis.png"), width=Inches(6.45)
    )
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "图 3  无权重 MLP 的 PR 曲线与一次完整 OOF 混淆矩阵")

    feature_rows = [
        [feature, f"{mean:.3f} ± {std:.3f}"]
        for feature, mean, std in top_features[:5]
    ]
    make_table(document, ["特征", "置换后 PR-AUC 下降"], feature_rows, [7.2, 7.2])
    add_caption(document, "表 4  无权重 MLP 的前五项置换特征重要性")
    add_body(
        document,
        "酒精度（0.181）、硫酸盐（0.141）和挥发性酸度（0.091）的置换损失最大，与图 1 的单变量关联方向基本一致；"
        "但密度、总二氧化硫等变量也具有不可忽略的联合贡献，说明模型并非只复现简单的 Pearson 排序。"
        "这些结果符合酿酒化学的一般认识：一定范围内酒精度与优质评价正相关，而过高的挥发性酸度通常对应不良风味；"
        "硫酸盐与抗氧化、发酵过程有关。该解释仅用于一致性检查，不能据此作因果推断。"
    )
    add_body(
        document,
        "方法优势有三点：实验规模小而完整，所有关键步骤均可复现；同时比较排序能力与阈值后决策结果；通过权重和阈值两个维度解释代价敏感学习。"
        "不足也很明确：数据只有 1599 条且来自特定地区，模型不能直接外推到其他产区；原始标签来自主观品评，存在噪声；"
        "重复交叉验证的折间结果相关，配对 t 检验只能作为描述性证据；单次混淆矩阵和图 1 的线性相关系数也都不代表因果关系。"
    )
    add_body(
        document,
        "后续可从三个方向改进：第一，按产区、年份或葡萄品种进行域外测试，并报告置信区间；第二，把二分类扩展为有序回归，利用质量等级的次序信息；"
        "第三，结合业务成本显式设置假阳性与假阴性代价，而不是预先指定类权重。若检测资源有限，还可学习排序分数并选择 top-k，"
        "这比固定阈值更贴近实际筛查流程。"
    )

    add_heading(document, "7  结论")
    add_body(
        document,
        "本文完成了一个可复现的优质红酒筛选小型研究。在严格隔离测试折的 5 折×3 种子评估中，小型 MLP 比逻辑回归取得更高的 Macro-F1 和 PR-AUC，"
        "证明 11 个理化指标间存在线性基线未充分捕获的非线性关系。代价敏感消融则表明：类权重能明显改变固定阈值下的召回/特异度权衡，"
        "但在阈值已经校准的条件下，继续增大权重并不会自动改善综合分类效果。该结论提醒实践者，在处理类别不平衡问题时，"
        "应把排序质量、阈值策略和业务代价分开验证。"
    )

    add_heading(document, "参考资料")
    references = [
        "[1] Cortez P., Cerdeira A., Almeida F., Matos T., Reis J. Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 2009, 47(4): 547–553.",
        "[2] UCI Machine Learning Repository. Wine Quality Dataset. https://archive.ics.uci.edu/dataset/186/wine+quality",
        "[3] Kingma D. P., Ba J. Adam: A Method for Stochastic Optimization. ICLR, 2015.",
        "[4] He K., Zhang X., Ren S., Sun J. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV, 2015.",
        "[5] Saito T., Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE, 2015.",
    ]
    for reference in references:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Pt(0)
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.paragraph_format.space_after = Pt(1)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = paragraph.add_run(reference)
        set_run_font(run, "宋体", 8.7)

    reproducibility = document.add_paragraph()
    reproducibility.paragraph_format.space_before = Pt(4)
    reproducibility.paragraph_format.space_after = Pt(0)
    run = reproducibility.add_run("复现与学术说明：")
    set_run_font(run, "黑体", 8.8, True)
    run = reproducibility.add_run(
        " 仓库包含原始数据、NumPy 训练脚本、完整 CSV 结果、绘图脚本和本报告生成脚本。"
        "数据与算法文献已在参考资料中说明；生成式 AI 仅用于代码组织、文字润色和排版辅助，全部实验数值均由本地脚本实际运行得到，未复制或编造他人结果。"
    )
    set_run_font(run, "宋体", 8.5)

    document.core_properties.title = "类别不平衡场景下基于多层感知机的优质红酒筛选"
    document.core_properties.subject = "深度学习课程设计 2026 秋季"
    document.core_properties.author = student_name
    document.core_properties.keywords = "类别不平衡, MLP, 代价敏感学习, 红酒质量"
    document.save(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", default="QSZ20262031")
    parser.add_argument("--student-name", default="黄海星")
    parser.add_argument("--github-url", default="")
    args = parser.parse_args()
    print(
        build_report(
            student_id=args.student_id,
            student_name=args.student_name,
            github_url=args.github_url,
        )
    )
