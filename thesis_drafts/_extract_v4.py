"""Extract docx to plain text (no line numbers)"""
from docx import Document
from docx.oxml.ns import qn

def iter_block_items(parent):
    body = parent.element.body
    for child in body:
        if child.tag == qn('w:p'):
            yield ('para', child)
        elif child.tag == qn('w:tbl'):
            yield ('table', child)

def extract_para_text(p_elem):
    parts = []
    for child in p_elem.iter():
        if child.tag == qn('w:t'):
            if child.text:
                parts.append(child.text)
        elif child.tag == qn('w:tab'):
            parts.append('\t')
        elif child.tag == qn('w:object'):
            parts.append('[公式]')
        elif 'OLEObject' in child.tag:
            parts.append('[公式]')
        elif child.tag == qn('m:oMath') or child.tag == qn('m:oMathPara'):
            math_text = ''.join(t.text or '' for t in child.iter(qn('m:t')))
            if math_text.strip():
                parts.append(f'[公式: {math_text.strip()}]')
            else:
                parts.append('[公式]')
        elif child.tag == qn('w:drawing') or child.tag == qn('w:pict'):
            parts.append('[图片]')
    return ''.join(parts)

def extract_table(tbl_elem):
    rows_data = []
    for tr in tbl_elem.iter(qn('w:tr')):
        cells = []
        for tc in tr.iter(qn('w:tc')):
            cell_text = []
            for p in tc.iter(qn('w:p')):
                t = extract_para_text(p).strip()
                if t:
                    cell_text.append(t)
            cells.append(' '.join(cell_text))
        rows_data.append(cells)
    if not rows_data:
        return ''
    lines = []
    for row in rows_data:
        lines.append(' | '.join(row))
    return '\n'.join(lines)

docx_path = r'C:\Users\caome\OneDrive\Desktop\论文\曹孟龙毕业论文_0413下午提意见改后.docx'
out_path = r'd:\repository\sealine_detection\thesis_drafts\thesis_full_text_v4.txt'
doc = Document(docx_path)
output = []
for block_type, elem in iter_block_items(doc):
    if block_type == 'para':
        text = extract_para_text(elem).strip()
        if text:
            output.append(text)
    elif block_type == 'table':
        output.append('[表格开始]')
        tbl_text = extract_table(elem)
        for tbl_line in tbl_text.split('\n'):
            if tbl_line.strip():
                output.append(tbl_line)
        output.append('[表格结束]')

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
print(f'Done: {len(output)} lines -> {out_path}')
