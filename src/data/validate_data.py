import great_expectations
import sys
import os
import pandas as pd
from definitions import PATH_TO_CURRENT_REFERENCE

def validate_is_active(has_failed):
    data_context = great_expectations.get_context()

    print("Validating data for is_active.csv")

    validate_data = data_context.run_checkpoint(
        checkpoint_name="is_active_checkpoint",
        batch_request=None,
        run_name=None
    )

    if not validate_data["success"]:
        has_failed = True
        print("Validation for is_active failed!")
        print("Errors:")
        for validation_result in validate_data["run_results"].values():
            for result in validation_result["validation_result"]["results"]:
                if not result["success"]:
                    print(result["expectation_config"]["kwargs"])
                    print(result["result"])
    else:
        print("Validation for is_active succeeded!")
    
    return has_failed


def validate_kudos(has_failed):

    data_context = great_expectations.get_context()

    print("Validating data for kudos.csv")

    validate_data = data_context.run_checkpoint(
        checkpoint_name="kudos_checkpoint",
        batch_request=None,
        run_name=None
    )

    if not validate_data["success"]:
        has_failed=True
        print("Validation for kudos failed!")
    else:
        print("Validation for kudos succeeded!")
    
    return has_failed


def main():

    has_failed = False

    # check if current or refernce data have the same length
    is_active_reference = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "is_active_reference.csv"))
    is_active_current = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "is_active_current.csv"))

    kudos_reference = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "kudos_reference.csv"))
    kudos_current = pd.read_csv(os.path.join(PATH_TO_CURRENT_REFERENCE, "kudos_current.csv"))

    if len(is_active_reference) != len(is_active_current):
        print("is_active has different length.. checking validation")        
        has_failed = validate_is_active(has_failed)
    else:
        print("is_active data has the same length in current and reference data.. skipping validation")

    if len(kudos_reference) != len(kudos_current):
        print("kudos data has different length in current and reference data")
        has_failed = validate_kudos(has_failed)
    else:
        print("kudos data has the same length in current and reference data.. skipping validation")

    if has_failed:
        print("Validation failed, exiting")
        sys.exit(1)
    else:
        print("Validation succeeded, exiting")        

if __name__ == '__main__':
    main()
