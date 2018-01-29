"""
(C) IBM Corp, 2018, All rights reserved
Created on Jan 09, 2018

@author: EHUD KARAVANI
"""
import os
import pandas as pd


INDEX_COL_NAME = "sample_id"
COUNTERFACTUAL_FILE_SUFFIX = "_cf"
FILENAME_EXTENSION = ".csv"
DELIMITER = ","


def combine_covariates_with_observed(covariate_file_path, factual_dir_path):
    """
    Python Generator to yield combined datasets. Will go over all factual files (i.e. tabular files containing the
    actual treatment assignment and the observed outcome for each sample) in a given directory, join them with the
    main covariate matrix and yield a full observed dataset.

    Args:
        covariate_file_path (str): Path to a file that is the main covariate matrix in a tabular format delimited by
                                   DELIMITER, having first row as a header row and a containing a INDEX_COL_NAME column
                                   (it serves as primary key to join on).
        factual_dir_path (str): Path to a directory containing factual file\s. Namely, a directory with many tabular
                                files delimited by DELIMITER, each containing a treatment assignment column (*z*), an
                                observed outcome column (*y*) and must contain a INDEX_COL_NAME column (as it serves as
                                primary key to join on).

    Yields:
        pd.DataFrame: An full observed dataset composed from a main covariate matrix and one factual file (i.e. a
                      treatment assignment and an observed outcome columns).
    """
    covariates = pd.DataFrame.from_csv(covariate_file_path, index_col=INDEX_COL_NAME, header=0, sep=DELIMITER)
    for file in os.listdir(factual_dir_path):
        if file.endswith(COUNTERFACTUAL_FILE_SUFFIX + FILENAME_EXTENSION):
            continue        # ignore in order to use only factual files
        factuals = pd.DataFrame.from_csv(os.path.join(factual_dir_path, file),
                                         index_col=INDEX_COL_NAME, header=0, sep=DELIMITER)
        dataset = covariates.join(factuals, how="inner")                                            # type: pd.DataFrame
        yield dataset
