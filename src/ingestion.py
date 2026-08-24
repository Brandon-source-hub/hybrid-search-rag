from pathlib import Path
from pypdf import PdfReader 

def load_txt(file_path):
    with open(file_path,"r",encoding="utf-8") as f:
        return f.read()
    

def load_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append(text)
    
    return "\n".join(texts)

def load_documents(directory):
    documents=[]
    directory = Path(directory)
    for file_path in directory.iterdir():
        if file_path.suffix.lower() == ".txt":
            text = load_txt(file_path)
            
        elif file_path.suffix.lower() == ".pdf":
            text = load_pdf(file_path)
        
        else:
            continue

        documents.append(
            {
                "source": file_path.name,
                "text": text
            }
        )
    
    return documents