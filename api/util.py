# Import necessary libraries
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rich.console import Console
from rich.table import Table
from rich import box

# Determine whether to use a GPU (CUDA) if available or fall back to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
console = Console()
console.print(f"[bold green]Using device: {device}[/bold green]")

# Load the Netflix dataset and remove entries with missing descriptions
df = pd.read_csv('netflix_titles.csv')
df = df.dropna(subset=['description']).reset_index(drop=True)

# Load the pre-trained Sentence Transformer model on the specified device
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

# Extract descriptions from the DataFrame and compute embeddings
descriptions = df['description'].tolist()
desc_embeddings = model.encode(descriptions, convert_to_tensor=True, device=device)

def recommend_movie(query, model, desc_embeddings, df, top_n=5):
    # Compute embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    
    # Calculate cosine similarity between the query and all movie descriptions
    cosine_scores = util.cos_sim(query_embedding, desc_embeddings)[0]
    
    # Move scores to CPU and convert to a numpy array for further processing
    cosine_scores_cpu = cosine_scores.cpu().numpy()
    
    # Get indices of the top_n most similar descriptions
    top_indices = np.argpartition(-cosine_scores_cpu, range(top_n))[:top_n]
    top_indices = top_indices[np.argsort(-cosine_scores_cpu[top_indices])]
    
    # Prepare a DataFrame with the top recommendations and compute match percentage
    recommended_df = df.iloc[top_indices][['title', 'description', 'listed_in', 'release_year']].copy()
    recommended_df['match_percentage'] = cosine_scores_cpu[top_indices] * 100
    return recommended_df

def display_recommendations(recommendations: pd.DataFrame):
    table = Table(
        title="[bold bright_blue]Recommended movies/TV Shows[/bold bright_blue]",
        title_style="bold underline",
        box=box.DOUBLE_EDGE,
        border_style="bright_green",
        show_lines=True,
        padding=(0, 1)
    )

    # Define table columns with styling
    table.add_column("Name", style="bold cyan", no_wrap=False)
    table.add_column("Description", style="green", no_wrap=False, overflow="fold", justify="left")
    table.add_column("Listed in", style="magenta", no_wrap=False)
    table.add_column("Release Year", style="yellow", no_wrap=True)
    table.add_column("Match", style="bright_red", no_wrap=True)

    # Add each recommendation as a row in the table
    for _, row in recommendations.iterrows():
        table.add_row(
            f"[bold]{row['title']}[/bold]",
            row['description'],
            row['listed_in'],
            str(row['release_year']),
            f"{row['match_percentage']:.2f}%"
        )

    return table