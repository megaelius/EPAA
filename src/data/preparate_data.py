import numpy as np
import pandas as pd
import math
import click
import logging
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):

    path1 = os.path.join(input_filepath,"GIT_COMMITS.csv")
    path2 = os.path.join(input_filepath,"SONAR_ANALYSIS.csv")
    path3 = os.path.join(input_filepath,"SONAR_MEASURES.csv")
    db_commits = pd.read_csv(path1,lineterminator='\n')
    db_analysis = pd.read_csv(path2)
    db_measures = pd.read_csv(path3)

    db_commits = db_commits[["PROJECT_ID", "COMMIT_HASH", "COMMIT_MESSAGE", "AUTHOR",  "COMMITTER_DATE"]]

    db_analysis = db_analysis[["REVISION", "ANALYSIS_KEY"]]

    db_measures = db_measures[["analysis_key", "complexity", "violations", "development_cost"]]
    db_measures = db_measures[db_measures["complexity"].notna()]

    db_sonar0 = db_analysis.merge(db_measures, left_on="ANALYSIS_KEY", right_on="analysis_key")
    db_sonar = db_sonar0.rename(columns = {'REVISION' : 'COMMIT_HASH'})
    db_sonar = db_sonar[["COMMIT_HASH", "complexity", "violations", "development_cost"]]

    db_merged = pd.merge(db_commits, db_sonar, how = 'left', on = 'COMMIT_HASH', indicator = True)

    db_merged = db_merged.sort_values(by=["PROJECT_ID","COMMITTER_DATE"]).reset_index()

    db_merged['inc_complexity'] = float("Nan")
    db_merged['inc_violations'] = float("Nan")
    db_merged['inc_development_cost'] = float("Nan")

    for i in range(1,db_merged.shape[0]):

        #first we make sure that both entries are from the same project (if not leave with the Nan value in the increment variable)
        if (db_merged['PROJECT_ID'][i] == db_merged['PROJECT_ID'][i-1]):

            for inc_variable in [["complexity",'inc_complexity'],["violations", "inc_violations"],["development_cost", "inc_development_cost"]]:
                variable_act = db_merged[inc_variable[0]][i] #value for the variable in the row i
                variable_past =  db_merged[inc_variable[0]][i-1] #value for the variable in the row before i

                if pd.notna(variable_act) and pd.notna(variable_past): #both entries available
                    db_merged[inc_variable[1]][i] = variable_act - variable_past
                else:
                    break

    db_increases = db_merged[db_merged['inc_complexity'].notna() & db_merged['inc_violations'].notna() & db_merged['inc_development_cost'].notna()]

    clean_db_merged = db_merged.dropna() #we delete al the NAs in the table
    final_db = clean_db_merged[["PROJECT_ID", "COMMIT_HASH", "COMMIT_MESSAGE", "AUTHOR", "COMMITTER_DATE", "inc_complexity", "inc_violations", "inc_development_cost"]]

    clean_column = [None]*len(final_db["COMMIT_MESSAGE"])
    for i in range(len(final_db["COMMIT_MESSAGE"])):
        clean = ''
        for j,s in enumerate(final_db["COMMIT_MESSAGE"].iloc[i].split()[:-3]):
            if j!=0:
                clean+= ' '
            clean+=s
        clean_column[i] = clean
    final_db["CLEAN_CMS"] = clean_column

    final_db.to_csv(os.path.join(output_filepath,'predictionDB.csv'), index='False') #export!

if __name__ == '__main__':
    main()
