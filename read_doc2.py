import docx
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

doc = docx.Document(r'e:\jianzi123.github.io\王帅俭_2026_0.docx')
for para in doc.paragraphs:
    if para.text.strip():
        print(para.text)
