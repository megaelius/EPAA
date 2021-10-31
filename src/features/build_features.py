import click
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')
@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())

def main(input_path, output_path):

    final_db = pd.read_csv(input_path + '/predictionDB.csv',lineterminator='\n')
    nrows = final_db.shape[0]

    if not Path(output_path + '/embeddings').exists():
        Path(output_path + '/embeddings').mkdir()

    if not Path(output_path + '/embeddings2').exists():
        Path(output_path + '/embeddings2').mkdir()

    for index,row in tqdm(final_db.iterrows()):
        frases = row['COMMIT_MESSAGE']
        matrix = model.encode(frases)
        np.save(output_path +'/embeddings/' + row['COMMIT_HASH'] + '.npy',matrix)

        frases = row['CLEAN_CMS']
        matrix = model.encode(frases)
        np.save(output_path +'/embeddings2/' + row['COMMIT_HASH'] + '.npy',matrix)


if __name__ == '__main__':
    main()
