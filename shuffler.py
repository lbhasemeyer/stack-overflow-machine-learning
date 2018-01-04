import numpy as np
import pandas as pd

def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index))


def main(outputfilename):
  shuffler('./survey_results_cut_down.csv').to_csv(outputfilename, sep=',')
  test_data = pd.read_csv('./' + outputfilename, header=0, nrows=10278)
  training_data = pd.read_csv('./' + outputfilename, header=0, skiprows=10278, nrows=41114)
  test_data.to_csv('test_data.csv', sep=',')
  training_data.to_csv('training_data.csv', sep=',')

if __name__ == '__main__':
  main('shuffled_data.csv')
