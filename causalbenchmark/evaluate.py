#!/usr/bin/env python
# Compatible with both python2 and python3
"""
A package and a command line script evaluating causal inference predictions.

(C) IBM Corp, 2017, All rights reserved
Created on Dec 19, 2017

@author: EHUD KARAVANI
"""
from __future__ import division as __division, print_function as __print_function
from future.utils import raise_with_traceback
import argparse
import os

import pandas as pd
import numpy as np


COUNTERFACTUAL_FILE_SUFFIX = "_cf"
FILENAME_EXTENSION = ".csv"
DELIMITER = ","

# Column header names from data
HEADER_Y1 = "y1"                            # cf files + individual prediction files: outcome under treatment
HEADER_Y0 = "y0"                            # cf files + individual prediction files: outcome under no treatment
HEADER_IND_IDX = "sample_id"                # cf files + individual prediction files: index column, sample id
HEADER_EFFECT_SIZE = "effect_size"          # prediction files + population prediction files: population effect size
HEADER_CI_LEFT = "li"           # prediction files + population prediction files: left confidence interval boundary
HEADER_CI_RIGHT = "ri"          # prediction files + population prediction files: right confidence interval boundary
HEADER_POP_IDX = "ufid"         # population prediction files: index column containing names of data instances

EPSILON = 1e-7                      # floating point baseline to avoid zero-division


def _score_population(predictions_location, cf_dir_location):
    """
    Scores estimations of treatment effect size over the population.

    Args:
        predictions_location (str): Path to a single tabular file where the effect estimations are located.
                                    Files must of tabular format
                                     * containing 4 columns: HEADER_POP_IDX, HEADER_EFFECT_SIZE, HEADER_CI_LEFT, HEADER_CI_RIGHT.
                                     * delimited by DELIMITER.
                                     * have FILENAME_EXTENSION extension to them.
                                    These global variables specified above can be changed when importing the module.
        cf_dir_location (str): Path to a directory containing the counter-factual files (i.e. labeled, ground-truth
                               data).
                               Files must be of tabular format
                                * containing 3 columns: HEADER_IND_IDX, HEADER_Y1, HEADER_Y0.
                                * delimited by DELIMITER.
                                * have the suffix specified in COUNTERFACTUAL_FILE_SUFFIX.
                                * have FILENAME_EXTENSION extension to them.
                               These global variables specified above can be changed when importing the module.

    Returns:
        pd.Series: Scores. Where Series' Index is the metric name and the value is the evaluation of that metric.
    """
    ufids = os.listdir(cf_dir_location)
    ufids = [f.rsplit("_")[0] for f in ufids if f.lower().endswith(COUNTERFACTUAL_FILE_SUFFIX + FILENAME_EXTENSION)]

    # Gather scoring statistics:
    ratio = pd.Series(index=ufids, name="population_ratio")
    bias = pd.Series(index=ufids, name="population_bias")
    ci_size = pd.Series(index=ufids, name="population_ci-size")
    coverage = pd.Series(data=False, index=ufids, dtype=np.dtype(bool), name="population_coverage")

    # Get data:      # HEADER_POP_IDX | HEADER_EFFECT_SIZE | HEADER_CI_LEFT | HEADER_CI_RIGHT
    estimates = pd.read_csv(predictions_location, index_col=HEADER_POP_IDX, sep=DELIMITER)
    assert set(estimates.index) == set(ufids)

    true_effects = pd.Series(index=ufids)
    dataset_sizes = pd.Series(index=ufids, name="size")

    for ufid in ufids:
        # Get the true effect:
        gt = pd.read_csv(os.path.join(cf_dir_location, ufid + COUNTERFACTUAL_FILE_SUFFIX + FILENAME_EXTENSION),
                         sep=DELIMITER)
        true_effect = np.mean(gt[HEADER_Y1] - gt[HEADER_Y0])
        true_effects[ufid] = true_effect

        # Get the population estimates:     | HEADER_EFFECT_SIZE | HEADER_CI_LEFT | HEADER_CI_RIGHT |
        estimate = estimates.loc[ufid, :]

        # Calculate the sufficient statistics:
        ratio[ufid] = (estimate[HEADER_EFFECT_SIZE] + EPSILON) / (true_effect + EPSILON)
        bias[ufid] = estimate[HEADER_EFFECT_SIZE] - true_effect
        ci_size[ufid] = estimate[HEADER_CI_RIGHT] - estimate[HEADER_CI_LEFT]            # right - left -> non-negative
        coverage[ufid] = estimate[HEADER_CI_LEFT] <= true_effect <= estimate[HEADER_CI_RIGHT]

        # Save the size of the current dataset:
        dataset_sizes[ufid] = gt.index.size
    dataset_sizes = dataset_sizes.astype(int)                                                       # type: pd.Series

    # Calculate metrics
    enormse = 1.0 - ratio                                                                           # type: pd.Series
    encis = ci_size / (true_effects.abs() + EPSILON)                                                # type: pd.Series
    cic = bias.abs() / ci_size
    # Aggregate by sizes:
    enormse_by_size = enormse.pow(2).groupby(by=dataset_sizes).mean().pow(0.5)
    rmse_by_size = bias.pow(2).groupby(by=dataset_sizes).mean().pow(0.5)
    bias_by_size = bias.groupby(by=dataset_sizes).mean()
    coverage_by_size = coverage.groupby(by=dataset_sizes).mean()
    encis_by_size = encis.groupby(by=dataset_sizes).mean()
    cic_by_size = cic.groupby(by=dataset_sizes).mean()

    results = pd.Series()
    if dataset_sizes.nunique() == 1:
        # return the by_sizes, they are enough since there's one size so just extract the scalar value they hold
        results["enormse"] = enormse_by_size.iloc[0]
        results["rmse"] = rmse_by_size.iloc[0]
        results["bias"] = bias_by_size.iloc[0]
        results["coverage"] = coverage_by_size.iloc[0]
        results["encis"] = encis_by_size.iloc[0]
        results["cic"] = cic_by_size.iloc[0]

    else:
        # weighted_sum = lambda x, w: x.mul(w).sum() / w.sum()
        def weighted_sum(x, w): return x.mul(w).sum() / w.sum()

        # Calculate the Weights for aggregation:
        weights = __get_weights(dataset_sizes)

        # Aggregate
        results["enormse"] = np.sqrt(weighted_sum(enormse_by_size.pow(2), weights))
        results["rmse"] = np.sqrt(weighted_sum(rmse_by_size.pow(2), weights))
        results["bias"] = weighted_sum(bias_by_size, weights)
        results["coverage"] = weighted_sum(coverage_by_size, weights)
        results["encis"] = weighted_sum(encis_by_size, weights)
        results["cic"] = weighted_sum(cic_by_size, weights)
        results = results.append(enormse_by_size.add_prefix("enormse_"))

    return results


def _score_individual(predictions_location, cf_dir_location):
    """
    Scores estimations of treatment effect size on individuals (i.e. the prediction of both outcome under no treatment
    and outcome under positive treatment for each individual).

    Args:
        predictions_location (str): Path to a directory containing tabular files with individual effect estimations
                               (i.e. prediction of factual and counterfactual outcomes for each individual).
                               Files must of tabular format
                                * containing 3 columns: HEADER_IND_IDX, HEADER_Y1, HEADER_Y0.
                                * delimited by DELIMITER.
                                * have FILENAME_EXTENSION extension to them.
                               These global variables specified above can be changed when importing the module.
        cf_dir_location (str): Path to a directory containing the counter-factual files (i.e. labeled, ground-truth
                               data).
                               Files must be of tabular format
                                * containing 3 columns: HEADER_IND_IDX, HEADER_Y1, HEADER_Y0.
                                * delimited by DELIMITER.
                                * have the suffix specified in COUNTERFACTUAL_FILE_SUFFIX.
                                * have FILENAME_EXTENSION extension to them.
                               These global variables specified above can be changed when importing the module.

    Returns:
        pd.Series: Scores. Where Series' Index is the metric name and the value is the evaluation of that metric.
    """
    ufids = os.listdir(cf_dir_location)
    ufids = [f.rsplit("_")[0] for f in ufids if f.lower().endswith(COUNTERFACTUAL_FILE_SUFFIX + FILENAME_EXTENSION)]

    enormse = pd.Series(index=ufids, name="_".join(["individual", "enormse"]))
    rmse = pd.Series(index=ufids, name="_".join(["individual", "rmse"]))
    bias = pd.Series(index=ufids, name="_".join(["individual", "bias"]))
    dataset_sizes = pd.Series(index=ufids, name="size")

    for ufid in ufids:
        # Get the true effect:
        gt = pd.read_csv(os.path.join(cf_dir_location, ufid + COUNTERFACTUAL_FILE_SUFFIX + FILENAME_EXTENSION),
                         index_col=HEADER_IND_IDX, sep=DELIMITER)
        true_effect = gt[HEADER_Y1] - gt[HEADER_Y0]

        # Get estimated effect:                         submission format:    N rows: patient_ID | Y0 | Y1
        estimates = pd.read_csv(os.path.join(predictions_location, ufid + FILENAME_EXTENSION),
                                index_col=HEADER_IND_IDX, sep=DELIMITER)
        estimated_effect = estimates[HEADER_Y1] - estimates[HEADER_Y0]

        # Evaluate:
        individual_bias = estimated_effect - true_effect
        bias[ufid] = individual_bias.mean()
        rmse[ufid] = individual_bias.pow(2).mean()
        # enormse[ufid] = np.mean((individual_bias) / (true_effect + EPSILON) ** 2)
        enormse[ufid] = np.mean((1 - ((estimated_effect + EPSILON) / (true_effect + EPSILON))) ** 2)

        # Save the size of the current dataset:
        dataset_sizes[ufid] = gt.index.size

    dataset_sizes = dataset_sizes.astype(int)                                                       # type: pd.Series
    enormse_by_size = enormse.groupby(by=dataset_sizes).mean().pow(0.5)
    rmse_by_size = rmse.groupby(by=dataset_sizes).mean().pow(0.5)
    bias_by_size = bias.groupby(by=dataset_sizes).mean()

    results = pd.Series()
    if dataset_sizes.nunique() == 1:
        results["enormse"] = enormse_by_size.iloc[0]
        results["rmse"] = rmse_by_size.iloc[0]
        results["bias"] = bias_by_size.iloc[0]

    else:
        weights = __get_weights(dataset_sizes)
        results["enormse"] = np.sqrt(enormse_by_size.pow(2).mul(weights).sum() / weights.sum())
        results["rmse"] = np.sqrt(rmse_by_size.pow(2).mul(weights).sum() / weights.sum())
        results["bias"] = bias_by_size.mul(weights).sum() / weights.sum()
        results = results.append(enormse_by_size.add_prefix("enormse_"))
    return results


def __get_weights(dataset_sizes):
    """
    Calculate weights for aggregating scores.
    Weights are calculated using the size of each dataset and how much instances are present for each size.

    Args:
        dataset_sizes (pd.Series): A vector depicting the of size of each dataset.
                                   {dataset_id -> n}  (where n is the dataset's size, i.e. number of samples)

    Returns:
        pd.Series: A vector the size of the unique datasets' sizes with each size and it's appropriate weight.
                   {n -> weight}
    """
    dataset_sizes_unique = dataset_sizes.unique()
    weights = pd.Series(data=dataset_sizes_unique, index=dataset_sizes_unique)  # Weigh the datasets' sizes
    weights *= dataset_sizes.value_counts()  # Weigh the amount of each size
    weights /= weights.min()  # re-scale to avoid inflated numbers
    return weights


def evaluate(predictions_location, cf_dir_location, is_individual_prediction=False):
    """
    Score a prediction against given counter-factual (i.e. labeled, ground-truth) data.

    Args:
        predictions_location (str): if prediction type is individual effect then a path to a directory containing
                                    estimations in tabular files for each data instance formatted as following:
                                     * containing 3 columns: HEADER_IND_IDX, HEADER_Y1, HEADER_Y0.
                                     * delimited by DELIMITER.
                                     * have FILENAME_EXTENSION extension to them.
                                    if prediction type is not individual (i.e. population effect), then a path to a
                                    single tabular file with each row corresponding to a data instance, formatted as
                                    following:
                                     * containing 4 columns: HEADER_IND_IDX, HEADER_EFFECT_SIZE, HEADER_CI_LEFT, HEADER_CI_RIGHT.
                                     * delimited by DELIMITER.
                                     * have FILENAME_EXTENSION extension to them.
                                    These global variables specified above can be changed when importing the module.
        cf_dir_location (str): Path to a directory containing the counter-factual files (i.e. labeled, ground-truth
                               data).
                               Files must be of tabular format
                                * containing 3 columns: *HEADER_IND_IDX*, *HEADER_Y1*, *HEADER_Y0*.
                                * delimited by DELIMITER.
                                * have the suffix specified in COUNTERFACTUAL_FILE_SUFFIX.
                                * have FILENAME_EXTENSION extension to them.
                               These global variables specified above can be changed when importing the module.
        is_individual_prediction (bool): whether prediction type is individual effect or population effect.

    Returns:
        pd.Series: Evaluation results of different metrics on the given prediction when comparing to the ground-truths
                   supplied.
                   Series' Index is the metric name and the value is the evaluation of that metric.
    """

    # Validate inputs:
    if is_individual_prediction:
        if not os.path.isdir(predictions_location):
            raise_with_traceback(RuntimeError("Individual prediction must be accompanied with a directory submission "
                                              "path containing the different submission files."))
    else:
        if not os.path.isfile(predictions_location):
            raise_with_traceback(RuntimeError("Population prediction must be accompanied with a path to a tabular file "
                                              "where each row corresponding to a different benchmark dataset."))

    # Score:
    if is_individual_prediction:
        scores = _score_individual(predictions_location, cf_dir_location)
    else:
        scores = _score_population(predictions_location, cf_dir_location)

    return scores


def __get_parser():
    parser = argparse.ArgumentParser(prog="Causal Inference Evaluation Script",
                                     description="Evaluating the performance of methods inferring causal effects from "
                                                 "observational data. \n"
                                                 "Evaluation can score estimations of treatment effect either on "
                                                 "the entire population or on individuals within it.")

    parser.add_argument("predictions_location",
                        help="Path to where prediction directory/file resides.\n"
                             "a tabular csv file is expected if prediction type is population and directory is "
                             "expected if prediction type is population. Refer the evaluate() docstring for more "
                             "information")
    parser.add_argument("cf_dir_location",
                        help="A path where the ground-truth data (i.e. counterfactual files) are located.\n",
                        type=str)
    parser.add_argument("-i", "--individual", dest="is_individual_prediction",
                        help="Whether to evaluate individual effect predictions If not stated, prediction type is "
                             "assumed to be of the treatment effect on the entire population.",
                        action="store_true")
    parser.add_argument("-o", "--output_path", dest="output_path",
                        help="If provided, a csv file of the evaluation results will be saved under path. If not, it "
                             "will be printed to terminal",
                        type=str)
    parser.add_argument("--cf_suffix", dest="cf_suffix",
                        help="If provided, a suffix distinguishing the counterfactual files from their corresponding "
                             "factual files (i.e. the files holding the actual treatment assignment and the observed"
                             "outcome). \n"
                             "For example, say the cf_suffix is '_cf', and say a factual (observed) data file is named "
                             "8c5f509.csv, then its corresponding counterfactual (unobserved) data file will be "
                             "8c5f509_cf.csv",
                        type=str, default=COUNTERFACTUAL_FILE_SUFFIX)
    return parser


def __main(argv):
    """
    Command-line main function.
    """
    # Score:
    scores = evaluate(argv.predictions_location, argv.cf_dir_location, argv.is_individual_prediction)

    # Output results
    if argv.output_path is not None:
        scores.to_csv(argv.output_path, header=True, index=True, encoding="utf-8", decimal=".", sep=DELIMITER)
    else:
        print("\n", scores)

    return scores


if __name__ == '__main__':
    argv = __get_parser().parse_args()
    COUNTERFACTUAL_FILE_SUFFIX = argv.cf_suffix
    __main(argv)
