import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz

def create_dataframe(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if not parts:
                continue
            user_id = parts[0]
            item_ids = parts[1:]
            for item_id in item_ids:
                data.append({'user_id': user_id, 'item_id': item_id})
    
    df = pd.DataFrame(data)
    return df

def create_interaction_matrix(df):
    user_ids = df['user_id'].values
    item_ids = df['item_id'].values
    values = np.ones(len(df))
    
    # Create a sparse matrix
    # Rows are users, columns are items
    interaction_matrix = csr_matrix((values, (user_ids, item_ids)))
    return interaction_matrix

if __name__ == "__main__":
    file_path = 'train-2.txt'
    df = create_dataframe(file_path)
    print(df.head())
    print(f"DataFrame shape: {df.shape}")
    
    interaction_matrix = create_interaction_matrix(df)
    print(f"Interaction Matrix shape: {interaction_matrix.shape}")
    print(f"Number of non-zero elements: {interaction_matrix.nnz}")
    
    save_npz('interaction_matrix.npz', interaction_matrix)
    print("Interaction matrix saved to interaction_matrix.npz")
