import json
import networkx as nx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import re

# Set stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Check networkx version
print(f"NetworkX version: {nx.__version__}")

# Load FAISS index and metadata
index = faiss.read_index("curriculum_index.faiss")
with open("curriculum_metadata.json", "r", encoding='utf-8') as f:
    metadata_store = json.load(f)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a multi-directed graph
G = nx.MultiDiGraph()

# Build the graph
for chunk in metadata_store:
    chunk_id = chunk["id"]
    learning_area = chunk["metadata"]["learning_area"]
    level = chunk["metadata"]["level"]
    section = chunk["metadata"]["section"]
    text = chunk["text"]

    G.add_node(learning_area, type="Learning Area")
    G.add_node(level, type="Level")
    G.add_node(section, type="Section")
    G.add_node(chunk_id, type="Chunk", text=text)

    G.add_edge(learning_area, level, relationship="has_level")
    G.add_edge(level, section, relationship="has_section")
    G.add_edge(section, chunk_id, relationship="contains_chunk")

# Print graph stats
print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print(f"Learning Areas: {len([n for n, d in G.nodes(data=True) if d['type'] == 'Learning Area'])}")
print(f"Levels: {len([n for n, d in G.nodes(data=True) if d['type'] == 'Level'])}")
print(f"Sections: {len([n for n, d in G.nodes(data=True) if d['type'] == 'Section'])}")
print(f"Chunks: {len([n for n, d in G.nodes(data=True) if d['type'] == 'Chunk'])}")

# Save the graph
graph_data = nx.node_link_data(G, edges="edges")
with open("curriculum_graph.json", "w", encoding='utf-8') as f:
    json.dump(graph_data, f)
print("Graph saved as curriculum_graph.json")

# Hybrid retrieval function
def query_graph_rag(query, top_k=5):
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        chunk = metadata_store[idx]
        chunk_id = chunk["id"]
        learning_area = chunk["metadata"]["learning_area"]
        level = chunk["metadata"]["level"]
        section = chunk["metadata"]["section"]
        text = chunk["text"]

        # Graph traversal
        related_nodes = []
        if G.has_node(learning_area):
            related_levels = [n for n in G.successors(learning_area) if G.nodes[n]["type"] == "Level" and n != "Unknown"]
            # Extract numeric level for sorting
            def get_level_num(lvl):
                single_match = re.search(r"Level (\d+)([A-D])?", lvl)
                paired_match = re.search(r"Levels (\d+) and (\d+)", lvl)
                by_end_match = re.search(r"By the end of Level (\d+)", lvl)
                
                if paired_match:
                    return (int(paired_match.group(1)) + int(paired_match.group(2))) / 2
                elif single_match:
                    num = int(single_match.group(1))
                    if single_match.group(2):
                        return num + (ord(single_match.group(2)) - ord('A') + 1) / 10
                    return num
                elif by_end_match:
                    return int(by_end_match.group(1))
                return float('inf') if "Foundation" in lvl else -1

            # Filter out "By the end of..." and "In Levels..." entirely, exclude current level
            related_levels = [lvl for lvl in related_levels if "By the end of" not in lvl and "In Levels" not in lvl and lvl != level]
            current_num = get_level_num(level)
            if current_num == -1:
                current_num = float('inf')
            related_levels = sorted(related_levels, key=lambda x: abs(get_level_num(x) - current_num))[:2]
            related_nodes.extend([(learning_area, lvl, "has_level") for lvl in related_levels])
        if G.has_node(level):
            related_sections = [n for n in G.successors(level) if G.nodes[n]["type"] == "Section" and n != section]
            related_nodes.extend([(level, sec, "has_section") for sec in related_sections])

        results.append({
            "chunk_id": chunk_id,
            "learning_area": learning_area,
            "level": level,
            "section": section,
            "text": text[:200] + "..." if len(text) > 200 else text,
            "related_nodes": related_nodes
        })
    
    return results

# Test query
query = "English Level 5"
results = query_graph_rag(query)
print(f"\nResults for query: '{query}'")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Learning Area: {result['learning_area']}")
    print(f"Level: {result['level']}")
    print(f"Section: {result['section']}")
    print(f"Text: {result['text']}")
    print("Related Nodes:")
    for parent, child, rel in result["related_nodes"]:
        print(f"  {parent} -> {child} ({rel})")