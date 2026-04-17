"""Extract text and tables from thesis docx, preserving document order."""
from docx import Document
from docx.oxml.ns import qn

docx_path = r"C:\Users\caome\OneDrive\Desktop\论文\曹孟龙毕业论文_0413下午提意见改后.docx"
output_path = r"d:\repository\sealine_detection\thesis_drafts\thesis_full_text_v3.txt"

doc = Document(docx_path)

lines = []

def extract_table(table):
    lines.append("[表格开始]")
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        lines.append(" | ".join(cells))
    lines.append("[表格结束]")

# Walk document body in order (paragraphs + tables interleaved)
for element in doc.element.body:
    tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
    if tag == "p":
        text = element.text or ""
        # Build full text from all runs
        full_text = ""
        for node in element.iter():
            ntag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
            if ntag == "t" and node.text:
                full_text += node.text
            elif ntag == "drawing" or ntag == "pict":
                full_text += "[图片]"
        full_text = full_text.strip()
        if full_text:
            lines.append(full_text)
    elif tag == "tbl":
        # Find matching Table object
        for table in doc.tables:
            if table._element is element:
                extract_table(table)
                break

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Done. Extracted {len(lines)} lines to {output_path}")
