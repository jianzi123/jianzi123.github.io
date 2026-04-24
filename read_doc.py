import docx
doc = docx.Document(r'e:\jianzi123.github.io\王帅俭_2026_0.docx')
for para in doc.paragraphs:
    if para.text.strip():
        print(repr(para.text))
