import PyPDF2
import sys

reader = PyPDF2.PdfReader(r'c:\Users\caome\OneDrive\Desktop\论文\曹孟龙毕业论文第一个方法.pdf')
print(f'Total pages: {len(reader.pages)}')

with open(r'd:\repository\sealine_detection\pdf_extracted.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            f.write(f'\n--- PAGE {i+1} ---\n')
            f.write(text)

print("Done! Written to pdf_extracted.txt")
