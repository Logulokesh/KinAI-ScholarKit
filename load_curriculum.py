from docx import Document
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import sys
import re

# Set stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()

# Initialize FAISS index
index = faiss.IndexFlatL2(embedding_dim)

# List of 25 documents
doc_files = [
    "English – Curriculum.docx", "Dance – Curriculum.docx", "Drama – Curriculum.docx",
    "Media Arts – Curriculum.docx", "Music – Curriculum.docx", "Visual Arts – Curriculum_0.docx",
    "Visual Communication Design – Curriculum.docx", "Health and Physical Education – Curriculum.docx",
    "Civics and Citizenship – Curriculum.docx", "Economics and Business – Curriculum.docx",
    "Geography – Curriculum.docx", "History - Curriculum_0.docx", "Chinese – Curriculum.docx",
    "French – Curriculum.docx", "German – Curriculum.docx", "Indonesian – Curriculum.docx",
    "Italian – Curriculum.docx", "Japanese – Curriculum.docx", "Korean – Curriculum.docx",
    "Modern Greek – Curriculum.docx", "Spanish – Curriculum.docx", "Mathematics – Curriculum.docx",
    "Science – Curriculum.docx", "Design and Technologies – Curriculum.docx",
    "Digital Technologies – Curriculum.docx"
]

# Chunking function with stricter level detection
def chunk_document(doc_path, learning_area):
    doc = Document(doc_path)
    chunks = []
    current_level = "Unknown"
    current_section = "Unknown"
    current_text = ""

    # Regex for level detection
    level_pattern = re.compile(r"^(Level \d+[A-Za-z]?|Levels \d+ and \d+|Foundation( Level [A-D])?|By the end of Level \d+|F–\d+|In Levels \d+ and \d+|From Foundation to Level \d+)", re.IGNORECASE)

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        # Strict level detection
        level_match = level_pattern.match(text)
        if level_match:
            if current_text:
                chunks.append({
                    "text": current_text,
                    "metadata": {"learning_area": learning_area, "level": current_level, "section": current_section}
                })
            current_level = level_match.group(0)
            current_section = "Level Description"
            current_text = text
        # Section detection
        elif any(keyword in text.lower() for keyword in ["achievement standard", "content description", "learning outcome", "curriculum"]):
            if current_text:
                chunks.append({
                    "text": current_text,
                    "metadata": {"learning_area": learning_area, "level": current_level, "section": current_section}
                })
            current_section = "Achievement Standard" if "achievement" in text.lower() else "Content Descriptions"
            current_text = text
        else:
            current_text += " " + text
    
    if current_text:
        chunks.append({
            "text": current_text,
            "metadata": {"learning_area": learning_area, "level": current_level, "section": current_section}
        })
    
    # Debug: Print chunk details
    print(f"Chunks for {os.path.basename(doc_path)}: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 for brevity
        try:
            print(f"  Chunk {i+1}: Level={chunk['metadata']['level']}, Section={chunk['metadata']['section']}, Text={chunk['text'][:50]}...")
        except UnicodeEncodeError:
            safe_text = chunk['text'][:50].encode('ascii', 'replace').decode('ascii')
            print(f"  Chunk {i+1}: Level={chunk['metadata']['level']}, Section={chunk['metadata']['section']}, Text={safe_text}...")
    return chunks

# Process all documents
folder_path = "data"
all_chunks = []
for doc_file in doc_files:
    learning_area = doc_file.replace(" – Curriculum.docx", "").replace("_0", "").replace(".docx_0", "")
    doc_path = os.path.join(folder_path, doc_file)
    if os.path.exists(doc_path):
        print(f"Processing: {doc_file}")
        chunks = chunk_document(doc_path, learning_area)
        all_chunks.extend(chunks)
    else:
        print(f"Warning: {doc_file} not found in 'data' folder.")

# Embed chunks and add to FAISS
embeddings = []
metadata_store = []
for i, chunk in enumerate(all_chunks):
    embedding = model.encode(chunk["text"])
    embeddings.append(embedding)
    metadata_store.append({"id": f"chunk_{i}", "text": chunk["text"], "metadata": chunk["metadata"]})

# Convert to numpy array and add to FAISS index
embeddings = np.array(embeddings).astype('float32')
index.add(embeddings)

# Save FAISS index and metadata
faiss.write_index(index, "curriculum_index.faiss")
with open("curriculum_metadata.json", "w", encoding='utf-8') as f:
    json.dump(metadata_store, f)

print(f"Loaded {len(all_chunks)} chunks into FAISS index.")

# Verification query
query = "English Level 1"
query_embedding = model.encode(query).astype('float32').reshape(1, -1)
distances, indices = index.search(query_embedding, 5)
print("\nTop 5 closest chunks (debug):")
for idx in indices[0]:
    chunk = metadata_store[idx]
    print(f"Learning Area: {chunk['metadata']['learning_area']}, Level: {chunk['metadata']['level']}, Section: {chunk['metadata']['section']}")
    print(f"Text: {chunk['text'][:100]}...\n")