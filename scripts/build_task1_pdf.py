"""Build the candidate handout: python -m scripts.build_task1_pdf.

Requires ReportLab (an optional authoring dependency, not needed by candidates).
Uses standard PDF fonts so no platform-specific font files are required.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/task-1-few-shot-classification.pdf"
INK = colors.HexColor("#17323C")
TEAL = colors.HexColor("#087F8C")
MUTED = colors.HexColor("#506671")
PALE = colors.HexColor("#F1F6F7")
LINE = colors.HexColor("#D8E4E7")
PAGE_W, PAGE_H = A4
LEFT = 44
WIDTH = PAGE_W - 2 * LEFT


def paragraph(pdf, text, x, top, width=WIDTH, size=10, leading=14, color=INK, bold=False):
    style = ParagraphStyle(
        "body", fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size, leading=leading, textColor=color, alignment=TA_LEFT,
    )
    content = Paragraph(text, style)
    _, height = content.wrap(width, PAGE_H)
    content.drawOn(pdf, x, top - height)
    return top - height


def section(pdf, number, title, top):
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(TEAL)
    pdf.drawString(LEFT, top, number)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(LEFT + 25, top, title)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4, invariant=1)
    pdf.setTitle("Task 1 - Few-shot classification as attention over a support set")
    pdf.setAuthor("Sebastian Stapf")
    pdf.setCreator("PhD interview handout generator")
    pdf.setSubject("PhD interview - Python and PyTorch candidate coding exercise")
    pdf.setFillColor(TEAL)
    pdf.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(LEFT, 798, "PHD CODING INTERVIEW  /  TASK 01")
    pdf.setFont("Helvetica", 8.5)
    pdf.drawRightString(PAGE_W - LEFT, 798, "PYTHON + PYTORCH  |  35-45 MIN")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 27)
    pdf.drawString(LEFT, 761, "Few-shot classification")
    pdf.setFont("Helvetica", 20)
    pdf.drawString(LEFT, 734, "as attention over a support set")
    paragraph(
        pdf,
        "Implement a classifier that uses attention over labelled support examples to classify "
        "queries. Focus on tensor indexing, matrix operations and numerical stability.",
        LEFT, 714, size=10.2, leading=14.5, color=MUTED,
    )
    pdf.setStrokeColor(LINE)
    pdf.line(LEFT, 671, PAGE_W - LEFT, 671)

    section(pdf, "01", "Episode inputs", 652)
    rows = [
        ["INPUT", "SHAPE / TYPE", "MEANING"],
        ["support_embeddings", "float [N_s, D]", "Support feature vectors"],
        ["support_labels", "int64 [N_s]", "Class IDs in [0, C)"],
        ["query_embeddings", "float [N_q, D]", "Query feature vectors"],
        ["query_labels", "int64 [N_q]", "Target class IDs in [0, C)"],
        ["num_classes", "integer C", "Total number of classes"],
        ["temperature", "float (default 0.1)", "Positive attention temperature"],
    ]
    table = Table(rows, colWidths=[147, 119, WIDTH - 266], rowHeights=[20] + [18] * 6)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("FONTNAME", (0, 1), (1, -1), "Courier"),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, LINE),
    ]))
    _, table_h = table.wrap(WIDTH, PAGE_H)
    table.drawOn(pdf, LEFT, 638 - table_h)

    section(pdf, "02", "For every query", 488)
    top = 472
    steps = [
        "Compute cosine similarity to every support example.",
        'Divide similarities by <font name="Courier">temperature</font>.',
        "Apply softmax over the support examples.",
        "Obtain class probabilities by summing the attention weights of support examples "
        "belonging to each class.",
        "Return the mean negative log-likelihood of the query labels and the predicted classes.",
    ]
    for index, text in enumerate(steps, 1):
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 9.5)
        pdf.drawString(LEFT + 2, top - 10, str(index))
        top = paragraph(pdf, text, LEFT + 20, top, WIDTH - 20, size=10, leading=13) - 5

    section(pdf, "03", "Implement", top - 18)
    code_top = top - 31
    code_height = 112
    pdf.setFillColor(PALE)
    pdf.roundRect(LEFT, code_top - code_height, WIDTH, code_height, 7, fill=1, stroke=0)
    pdf.setFillColor(TEAL)
    pdf.rect(LEFT, code_top - code_height + 10, 3, code_height - 20, fill=1, stroke=0)
    code = [
        "def matching_loss(",
        "    support_embeddings,",
        "    support_labels,",
        "    query_embeddings,",
        "    query_labels,",
        "    num_classes,",
        "    temperature=0.1,",
        "):",
        "    ...",
    ]
    pdf.setFont("Courier", 9.2)
    for index, line in enumerate(code):
        pdf.setFillColor(TEAL if index == 0 else INK)
        pdf.drawString(LEFT + 15, code_top - 16 - index * 10.5, line)
    bottom = paragraph(
        pdf, '<b>Return:</b> <font name="Courier">(loss, predictions)</font> - a scalar mean NLL '
        'and <font name="Courier">LongTensor[N_q]</font> of highest-probability class IDs. '
        'Keep the loss differentiable with respect to both embedding tensors.',
        LEFT, code_top - code_height - 10, size=9.5, leading=13,
    )

    bottom = paragraph(
        pdf, "<b>Requirements.</b> Do not assume examples are grouped by class or that classes "
        "have equal support counts. Use vectorized, numerically stable PyTorch operations. "
        "Treat a zero embedding as a zero normalized vector.",
        LEFT, bottom - 13, size=9.5, leading=13,
    )
    bottom = paragraph(
        pdf, "<b>Assume valid inputs:</b> non-empty sets, a shared positive embedding dimension, "
        "finite floating-point embeddings of the same dtype, tensors on one device, valid int64 "
        "labels, C &gt; 0 and a finite positive temperature. Every query target class occurs in "
        "the support set; other classes may be absent and receive zero probability. "
        "Input validation is not required.",
        LEFT, bottom - 7, size=8.9, leading=12,
    )
    bottom = paragraph(
        pdf, "Be ready to explain your shapes, numerical choices, complexity and tests. "
        "No training or dataset download is needed.",
        LEFT, bottom - 8, size=8.9, leading=12, color=MUTED,
    )
    if bottom < 55:
        raise RuntimeError(f"Content overlaps footer: bottom={bottom:.1f}")
    pdf.setStrokeColor(LINE)
    pdf.line(LEFT, 40, PAGE_W - LEFT, 40)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.4)
    pdf.drawString(LEFT, 26, "CANDIDATE HANDOUT  /  Few-shot classification")
    pdf.drawRightString(PAGE_W - LEFT, 26, "TASK 1  /  PAGE 1 OF 1")
    pdf.showPage()
    pdf.save()
    print(f"Created {OUTPUT}; content ends at {bottom:.1f} pt")


if __name__ == "__main__":
    main()
