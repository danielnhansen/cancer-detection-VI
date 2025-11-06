"""
The following is a simple example evaluation method.

It is meant to run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the evaluation and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""
import json
import csv
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from pathlib import Path
from pprint import pformat, pprint
from helpers import run_prediction_processing, tree

INPUT_DIRECTORY = Path("./input")
print('input dir', INPUT_DIRECTORY)
OUTPUT_DIRECTORY = Path("./output")
print('output dir', OUTPUT_DIRECTORY)
GROUND_TRUTH_DIRECTORY = Path("./ground_truth")
print('ground_truth dir', GROUND_TRUTH_DIRECTORY)



def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any specific order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently
    results = run_prediction_processing(fn=process, predictions=predictions)
    # One row per study per breast
    flatten_results = []
    for item in results:
        flatten_results.append(item[0])
        flatten_results.append(item[1])
    df_results = pd.DataFrame(flatten_results)
    
    print(df_results)
    
    # Convert true labels into one-hot encoded format
    y_true = label_binarize(df_results['label'], classes=['0', '1', '2'])
    print(y_true)
    y_pred = df_results[['normal', 'benign', 'malignant']].values
    print(y_pred)
    
    # Compute micro-averaged ROC curve
    fpr, tpr, thresholds = roc_curve(y_true.ravel(), y_pred.ravel(), drop_intermediate=False)
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    print(thresholds)
    print(tpr)
    print(fpr)

    # Sensitivity at 90% specificity
    specificity_threshold = 0.90
    fpr_threshold = 1 - specificity_threshold
    sensitivity_at_90_specificity = np.interp(fpr_threshold, fpr, tpr)
    

    # Specificity at 90% sensitivity
    sensitivity_threshold = 0.90
    fpr_at_90_sensitivity = np.interp(sensitivity_threshold, tpr, fpr)
    specificity_at_90_sensitivity = 1 - fpr_at_90_sensitivity
    
    amalgamated_results = [roc_auc, specificity_at_90_sensitivity, sensitivity_at_90_specificity]
    averaged_results = np.mean(amalgamated_results)

    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission
    metrics["results"] = {
        "AUC": roc_auc,
        "Specificity": specificity_at_90_sensitivity,
        "Sensitivity": sensitivity_at_90_specificity,
        "Score": averaged_results
    }
    
    print('AUC:', roc_auc)
    print('Specificity:', specificity_at_90_sensitivity)
    print('Sensitivity:', sensitivity_at_90_specificity)
    print('Score:', averaged_results)

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0

def process(job):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    
    location_bilateral_breast_classification_likelihoods = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="bilateral-breast-classification-likelihoods",
    )
    print(location_bilateral_breast_classification_likelihoods)

    # Secondly, read the results
    result_bilateral_breast_classification_likelihoods = load_json_file(
        location=location_bilateral_breast_classification_likelihoods,
    )
    
    left_results = result_bilateral_breast_classification_likelihoods['left']
    print('left results\n',left_results)
    
    right_results = result_bilateral_breast_classification_likelihoods['right']
    print('right results\n',right_results)

    # Thirdly, retrieve the input file name to match it with your ground truth
    image_name_dce_breast_mri_timepoint_0 = get_image_name(
        values=job["inputs"],
        slug="dce-breast-mri-timepoint-0",
    )
    # All filenames of uploaded cases in the format "studyID^scan_name.nii"
    studyID = image_name_dce_breast_mri_timepoint_0.split('^')[0]

    # Fourthly, load your ground truth
    ground_truth_dir = GROUND_TRUTH_DIRECTORY
    
    # Ground truth csv headers "studyID","Lesion_Left","Lesion_Right"
    path_to_ground_truth = Path(ground_truth_dir,"pre-evaluation_ground_truth.csv")
    truth_dict = load_csv_to_dict(path_to_ground_truth)
    
    report += studyID
    
    report += 'Left prediction:'
    for classification in list(left_results):
        report += classification + ': ' + str(left_results[classification])
    report += 'Left truth:' + truth_dict[studyID][0]
    
    report += 'Right prediction:'
    for classification in list(right_results):
        report += classification + ': ' + str(right_results[classification])
    report += 'Left truth:' + truth_dict[studyID][1]
    
    print(report)
    
    results = []
    
    results.append({
        "ID": studyID,
        "normal": left_results['normal'],
        "benign": left_results['benign'],
        "malignant": left_results['malignant'],
        "label": truth_dict[studyID][0],
    })
    results.append({
        "ID": studyID,
        "normal": right_results['normal'],
        "benign": right_results['benign'],
        "malignant": right_results['malignant'],
        "label": truth_dict[studyID][1],
    })
    
    print(studyID, '\n', results)
    
    # For now, we will just report back some bogus metric
    return results


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    print("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        print(line)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


# def get_interface_key(job):
    # # Each interface has a unique key that is the set of socket slugs given as input
    # socket_slugs = [sv["interface"]["slug"] for sv in job["inputs"]]
    # return tuple(sorted(socket_slugs))


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        #print(value["interface"]["slug"])
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
        
def load_csv_to_dict(csv_filepath):
    dictionary = {}
    with open(csv_filepath, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for row in reader:
            study = row[0]
            left = row[1]
            right = row[2]
            dictionary[study] = (left, right)
    return dictionary


if __name__ == "__main__":
    print('point 1')
    raise SystemExit(main())
