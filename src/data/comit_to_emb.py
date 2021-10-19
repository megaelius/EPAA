import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


final_db = pd.read_csv('data/processed/predictionDB.csv',lineterminator='\n')

nrows = final_db.shape[0]

for index,row in tqdm(final_db.iterrows()):
    frases = row['COMMIT_MESSAGE']
    matrix = model.encode(frases)
    np.save(f'data/processed/embeddings/' + row['COMMIT_HASH'] + '.npy',matrix)
