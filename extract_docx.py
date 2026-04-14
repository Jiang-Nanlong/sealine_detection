"""Extract text from thesis docx file."""
import sys
try:
    from docx import Document
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    from docx import Document

docx_path = r"C:\Users\caome\OneDrive\Desktop\论文\曹孟龙毕业论文（精简版）.docx"
output_path = r"d:\repository\sealine_detection\thesis_drafts\thesis_full_text_v2.txt"

doc = Document(docx_path)

lines = []
for para in doc.paragraphs:
    text = para.text.strip()
    if text:
        lines.append(text)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Done. Extracted {len(lines)} lines to {output_path}")
