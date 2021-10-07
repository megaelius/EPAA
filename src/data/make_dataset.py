# -*- coding: utf-8 -*-
import os
import click
import logging
import sqlite3
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #create output folder
    if not Path(output_filepath).exists():
        Path(output_filepath).mkdir()

    # Create your connection.
    con = sqlite3.connect(input_filepath)

    # creating cursor
    cur = con.cursor()

    # reading all table names
    table_list = [a[0] for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    # here is you table list
    print(table_list)

    for table in tqdm(table_list):
        df = pd.read_sql_query("SELECT * FROM " + table, con)
        df.to_csv(os.path.join(output_filepath,table+'.csv'),index = False)

    # Be sure to close the connection
    con.close()




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
