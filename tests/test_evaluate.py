"""
(C) IBM Corp, 2018, All rights reserved
Created on Jan 10, 2018

@author: EHUD KARAVANI
"""
from __future__ import division as __division
import os
import unittest
import pandas as pd
import numpy as np

from causalbenchmark import evaluate
evaluate.TABULAR_DELIMITER = "\t"


class TestEvaluate(unittest.TestCase):
    def test_individual_single_size(self):
        # Right prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "single_sized_datasets",
                                               "individual_prediction_right"),
                                  os.path.join("test_data_files", "single_sized_datasets", "dummy_data"),
                                  is_individual_prediction=True)
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[0.0, 0.0, 0.0], index=["enormse", "rmse", "bias"]))

        # Wrong Prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "single_sized_datasets",
                                               "individual_prediction_wrong"),
                                  os.path.join("test_data_files", "single_sized_datasets", "dummy_data"),
                                  is_individual_prediction=True)
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[np.sqrt(17.0/9.0), np.sqrt((4 + 1 + (2/75)) / 3),
                                                               (-2 - 1 + (2.0 / 15)) / 3],
                                                         index=["enormse", "rmse", "bias"]))

    def test_population_single_size(self):
        # Right prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "single_sized_datasets",
                                               "population_prediction_right.csv"),
                                  os.path.join("test_data_files", "single_sized_datasets", "dummy_data"),
                                  is_individual_prediction=False)
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[0.0, 0.0, 0.0, 1.0, 5.0/3.0, 0.0],
                                                         index=["enormse", "rmse", "bias", "coverage", "encis", "cic"]))

        # Wrong Prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "single_sized_datasets",
                                               "population_prediction_wrong.csv"),
                                  os.path.join("test_data_files", "single_sized_datasets", "dummy_data"),
                                  is_individual_prediction=False)
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[np.sqrt(5.0/12.0), np.sqrt((0.25 + 0 + 0.04) / 3),
                                                               (-0.5 + 0 + 0.2) / 3,
                                                               2.0/3.0, 5.0/3.0, 5.0/12.0],
                                                         index=["enormse", "rmse", "bias", "coverage", "encis", "cic"]))

    def test_individual_multi_size(self):
        # Right prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "multi_sized_datasets",
                                               "individual_prediction_right"),
                                  os.path.join("test_data_files", "multi_sized_datasets", "dummy_data"),
                                  is_individual_prediction=True)
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[0.0, 0.0, 0.0, 0.0, 0.0],
                                                         index=["enormse", "rmse", "bias", "enormse_4", "enormse_6"]))

        # Wrong Prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "multi_sized_datasets",
                                               "individual_prediction_wrong"),
                                  os.path.join("test_data_files", "multi_sized_datasets", "dummy_data"),
                                  is_individual_prediction=True)
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[np.sqrt(((5.0 / 2.0) + (2.25 * 17.0 / 9.0)) / 3.25),
                                                               np.sqrt(((4*(4 + 0.04)) + 6*(4 + 1 + (4.0 / 150))) / 26),
                                                               (4*(-2 + 0.2) + 6*(-2 - 1 + (2.0/15))) / (4*2 + 6*3),
                                                               np.sqrt(5.0 / 2.0), np.sqrt(17.0/9.0)],
                                                         index=["enormse", "rmse", "bias", "enormse_4", "enormse_6"]))

    def test_population_multi_size(self):
        # Right prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "multi_sized_datasets",
                                               "population_prediction_right.csv"),
                                  os.path.join("test_data_files", "multi_sized_datasets", "dummy_data"),
                                  is_individual_prediction=False)
        score = score[["enormse", "enormse_4", "enormse_6", "rmse", "bias", "coverage", "encis", "cic"]]  # rearrange
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                               (6 * (2 + 2 + 1) + 4 * (2 + 1)) / (6 * 3 + 4 * 2),
                                                               0.0],
                                                         index=["enormse", "enormse_4", "enormse_6", "rmse", "bias",
                                                                "coverage", "encis", "cic"]))

        # Wrong Prediction:
        score = evaluate.evaluate(os.path.join("test_data_files", "multi_sized_datasets",
                                               "population_prediction_wrong.csv"),
                                  os.path.join("test_data_files", "multi_sized_datasets", "dummy_data"),
                                  is_individual_prediction=False)
        score = score[["enormse", "enormse_4", "enormse_6", "rmse", "bias", "coverage", "encis", "cic"]]  # rearrange
        np.testing.assert_array_almost_equal(x=score,
                                             y=pd.Series(data=[np.sqrt(((4*(0.25+1)) + (6*(0.25+0+1))) / (4*2 + 6*3)),
                                                               np.sqrt((0.25 + 1) / 2), np.sqrt((1 + 0 + 0.25) / 3),
                                                               np.sqrt((4*(0.25+0.04) + 6*(0.25+0+0.04)) / (4*2 + 6*3)),
                                                               (4*(-0.5 + 0.2) + 6*(-0.5 + 0 + 0.2)) / (4*2 + 6*3),
                                                               (4 * 1 + 6 * 2) / (4 * 2 + 6 * 3),
                                                               (6 * (2 + 2 + 1) + 4 * (2 + 1)) / (6 * 3 + 4 * 2),
                                                               (4 * (0.25 + 1) + 6 * (0.25 + 0 + 1)) / (4 * 2 + 6 * 3)],
                                                         index=["enormse", "enormse_4", "enormse_6", "rmse", "bias",
                                                                "coverage", "encis", "cic"]))

    def test_input_consistency(self):
        data_path = os.path.join("test_data_files", "multi_sized_datasets", "dummy_data")

        # population effect with directory input:
        dir_path = os.path.join("test_data_files", "multi_sized_datasets", "individual_prediction_right")
        is_individual_prediction = False
        with self.subTest(dir_path=dir_path, data_path=data_path, is_individual_prediction=is_individual_prediction):
            self.assertRaises(RuntimeError, evaluate.evaluate,
                              dir_path, data_path, is_individual_prediction)

        # individual effect with file input:
        file_path = os.path.join("test_data_files", "multi_sized_datasets", "population_prediction_wrong.csv")
        is_individual_prediction = True
        with self.subTest(file_path=file_path, data_path=data_path, is_individual_prediction=is_individual_prediction):
            self.assertRaises(RuntimeError, evaluate.evaluate,
                              file_path, data_path, is_individual_prediction)

    def test_missing_predictions(self):
        data_path = os.path.join("test_data_files", "single_sized_datasets", "dummy_data")
        individual_prediction_dir_path = os.path.join("test_data_files", "single_sized_datasets",
                                                      "individual_prediction_missing")
        self.assertRaises(IOError, evaluate._score_individual, individual_prediction_dir_path, data_path)

        population_prediction_file_path = os.path.join("test_data_files", "single_sized_datasets",
                                                       "population_prediction_missing.csv")
        self.assertRaises(AssertionError, evaluate._score_population, population_prediction_file_path, data_path)


if __name__ == '__main__':
    unittest.main()
