"""从原始PDF提取第四章实验部分文本，重点关注退化鲁棒性实验"""
import fitz  # pymupdf
import re

pdf_path = r"C:\Users\caome\OneDrive\Desktop\论文\曹孟龙毕业论文第二个方法.pdf"

doc = fitz.open(pdf_path)
print(f"Total pages: {doc.page_count}")

# 提取全文
all_text = []
for i, page in enumerate(doc):
    text = page.get_text()
    all_text.append(f"\n===== PAGE {i+1} =====\n{text}")

full = '\n'.join(all_text)

# 输出到文件
out_path = r"d:\repository\sealine_detection\thesis_drafts\method2_original_pdf.txt"
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(full)
print(f"Saved: {out_path} ({len(full)} chars)")
