import PyPDF2

reader = PyPDF2.PdfReader(r'c:\Users\caome\OneDrive\Desktop\吴迪硕士论文.pdf')
print(f'Total pages: {len(reader.pages)}')

with open(r'd:\repository\sealine_detection\pdf_wudi.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            f.write(f'\n--- PAGE {i+1} ---\n')
            f.write(text)

print("Done!")
