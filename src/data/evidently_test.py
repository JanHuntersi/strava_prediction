from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.report import Report
from evidently.tests import *
import sys
import os
import pandas as pd
import warnings
from definitions import PATH_TO_CURRENT_REFERENCE,PATH_TO_REPORTS_EVIDENTLY

def run_evidently_test(reference_data, current_data, report_name,results_name):

    # create a test suite
    test_suite = TestSuite([ TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])

    # calculate the test results
    results = test_suite.run(reference_data=reference_data, current_data=current_data)

    # create a report
    report = Report(results)

    # save the report
    report.save(os.path.join(PATH_TO_REPORTS_EVIDENTLY, report_name))

    test_suite.save_html(os.path.join(PATH_TO_REPORTS_EVIDENTLY, results_name))

    print("Data validation completed. Reports saved to current_reference folder")


def data_drift(reference_data, current_data, report_name):

    data_drift_preset_report = Report(
        metrics=[DataDriftPreset()]
    )

    data_drift_preset_report.run(reference_data=reference_data, current_data=current_data)

    data_drift_preset_report.save_html(os.path.join(PATH_TO_REPORTS_EVIDENTLY, report_name))

    print(f"Data drift report successfully saved to file {report_name} ")


def main():

    warnings.filterwarnings("ignore")

    # check if current or refernce data have the same length
    is_active_reference = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "is_active_reference.csv"))
    is_active_current = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "is_active_current.csv"))

    kudos_reference = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "kudos_reference.csv"))
    kudos_current = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "kudos_current.csv"))

    if len(is_active_reference) != len(is_active_current):
        print("Data length for is_active is different between current and reference data")

        data_drift(is_active_reference, is_active_current, "is_active_drift_report.html")

        run_evidently_test(is_active_reference, is_active_current, "is_active_report.html","index.html")

    else:
        print("Data length for is_active is the same between current and reference data .... skipping test")
     


    if len(kudos_reference) != len(kudos_current):
        print("Data length for kudos is different between current and reference data")
        
        data_drift(kudos_reference, kudos_current, "kudos_drift_report.html")

        run_evidently_test(kudos_reference, kudos_current, "kudos_report.html","kudos_results.html")
    else:
        print("Data length for kudos is the same between current and reference data .... skipping test")

    

if __name__ == "__main__":
    main()
