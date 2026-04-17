"""extract_docx_full.py — 从 docx 提取全文（含表格、公式占位）"""
import sys
from docx import Document
from docx.oxml.ns import qn
import re

def iter_block_items(parent):
    """按文档顺序迭代段落和表格"""
    body = parent.element.body
    for child in body:
        if child.tag == qn('w:p'):
            yield ('para', child)
        elif child.tag == qn('w:tbl'):
            yield ('table', child)

def extract_para_text(p_elem):
    """从段落XML提取文本，遇到OLE对象/图片标记占位"""
    parts = []
    for child in p_elem.iter():
        if child.tag == qn('w:t'):
            if child.text:
                parts.append(child.text)
        elif child.tag == qn('w:tab'):
            parts.append('\t')
        # MathType OLE 对象
        elif child.tag == qn('w:object'):
            parts.append('[公式]')
        elif 'OLEObject' in child.tag:
            parts.append('[公式]')
        # OMML 内嵌数学公式
        elif child.tag == qn('m:oMath') or child.tag == qn('m:oMathPara'):
            math_text = ''.join(t.text or '' for t in child.iter(qn('m:t')))
            if math_text.strip():
                parts.append(f'[公式: {math_text.strip()}]')
            else:
                parts.append('[公式]')
        # 图片
        elif child.tag == qn('w:drawing') or child.tag == qn('w:pict'):
            parts.append('[图片]')
    return ''.join(parts)

def extract_table(tbl_elem):
    """提取表格为文本格式"""
    doc_tbl = None
    # 需要通过 Document 的 Table 包装
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
    
    # 格式化为对齐表格
    lines = []
    for row in rows_data:
        lines.append(' | '.join(row))
    return '\n'.join(lines)

def main():
    docx_path = r"C:\Users\caome\OneDrive\Desktop\论文\曹孟龙毕业论文_0413下午提意见改后.docx"
    out_path = r"d:\repository\sealine_detection\thesis_drafts\thesis_full_text_v3.txt"
    
    print(f"Loading: {docx_path}")
    doc = Document(docx_path)
    
    output = []
    line_num = 0
    
    for block_type, elem in iter_block_items(doc):
        if block_type == 'para':
            text = extract_para_text(elem).strip()
            if text:
                line_num += 1
                output.append(f"L{line_num:04d} | {text}")
        elif block_type == 'table':
            line_num += 1
            output.append(f"L{line_num:04d} | [表格开始]")
            tbl_text = extract_table(elem)
            for tbl_line in tbl_text.split('\n'):
                if tbl_line.strip():
                    line_num += 1
                    output.append(f"L{line_num:04d} | {tbl_line}")
            line_num += 1
            output.append(f"L{line_num:04d} | [表格结束]")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f"Done: {len(output)} lines -> {out_path}")

if __name__ == '__main__':
    main()
