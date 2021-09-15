import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "CMU-MultimodalSDK"))

from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import log


feature_selection = {
    "cmumosei": {
        "audio": "COVAREP",
        "text": "glove_vectors",
        "visual": "FACET 4.2",
        "labels": "All Labels",
    },
    "cmumosi": {
        "audio": "COVAREP",
        "text": "glove_vectors",
        "visual": "FACET_4.2",
        "labels": "Opinion Segment Labels",
    },
    "pom": {
        "audio": "COVAREP",
        "text": "glove_vectors",
        "visual": "FACET 4.2",
        "labels": "labels"
    }
}

dataset_configs = {
    "cmumosei": mmdatasdk.cmu_mosei,
    "cmumosi": mmdatasdk.cmu_mosi,
    "pom": mmdatasdk.pom
}


def deploy(in_dataset,destination):
    deploy_files={x:x for x in in_dataset.keys()}
    in_dataset.deploy(destination,deploy_files)


def download_data(dataset_config, dataset_name):
    source = {}
    dataset = {}
    dataset_path = os.path.join("data", dataset_name)
    folder_path = os.path.join(dataset_path, dataset_name)

    if not os.path.isdir(folder_path + "_raw"):
        source["raw"] = {'words': dataset_config.raw["words"]}
    else:
        print("Raw already exist, no need to download")
        dataset["raw"] = mmdatasdk.mmdataset(folder_path + "_raw")

    if not os.path.isdir(folder_path + "_highlevel"):
        tmp_dict = feature_selection[dataset_name]
        del tmp_dict["labels"]
        highlevel = {tmp_dict[key]: dataset_config.highlevel[tmp_dict[key]] for key in tmp_dict}
        source["highlevel"] = highlevel
    else:
        print("Highlevel features already exist, no need to download")
        dataset["highlevel"] = mmdatasdk.mmdataset(folder_path + "_highlevel")

    if not os.path.isdir(folder_path + "_labels"):
        source["labels"] = dataset_config.labels
    else:
        print("Labels already exist, no need to download")
        dataset["labels"] = mmdatasdk.mmdataset(folder_path + "_labels")

    is_empty = not source
    if not is_empty:
        for key in source:
            dataset[key] = mmdatasdk.mmdataset(source[key], folder_path + "_" + key)


def process_sequences(dataset_name, seq_len=50):

    dataset_path = os.path.join("data", dataset_name)
    folder_path = os.path.join(dataset_path, dataset_name)

    dataset = {}

    dataset["labels"] = mmdatasdk.mmdataset(folder_path + "_labels")

    word_aligned_path = folder_path + '_word_aligned_highlevel'
    final_aligned_path = folder_path + '_final_aligned_highlevel'

    if os.path.isdir(final_aligned_path) and os.listdir(final_aligned_path) != []:
        dataset["highlevel"] = mmdatasdk.mmdataset(final_aligned_path)

    else:
        if os.path.isdir(word_aligned_path) and os.listdir(word_aligned_path) != []:
            dataset["highlevel"] = mmdatasdk.mmdataset(word_aligned_path)

        else:
            dataset["highlevel"] = mmdatasdk.mmdataset(folder_path + "_highlevel")

            dataset["highlevel"].align("glove_vectors")
            dataset["highlevel"].impute('glove_vectors')
            deploy(dataset["highlevel"], word_aligned_path)

        dataset["highlevel"].computational_sequences[feature_selection[dataset_name]["labels"]] = \
            dataset["labels"][feature_selection[dataset_name]["labels"]]

        dataset["highlevel"].align(feature_selection[dataset_name]["labels"])
        dataset["highlevel"].hard_unify()
        deploy(dataset["highlevel"], final_aligned_path)

    non_sequences = [feature_selection[dataset_name]["labels"]]

    tensors = dataset["highlevel"].get_tensors(
        seq_len=seq_len, non_sequences=non_sequences, direction=False,
        folds=[dataset_configs[dataset_name].standard_folds.standard_train_fold,
               dataset_configs[dataset_name].standard_folds.standard_valid_fold,
               dataset_configs[dataset_name].standard_folds.standard_test_fold])

    fold_names = ["train", "valid", "test"]

    for i in range(3):
        for csd in list(dataset["highlevel"].keys()):
            print("Shape of the %s computational sequence for %s fold is %s"%(
                csd, fold_names[i], tensors[i][csd].shape))

    return tensors


def get_and_process_data(dataset_name):

    download_data(dataset_configs[dataset_name], dataset_name)
    tensors = process_sequences(dataset_name)
    log.success("Dataset processed")
    return tensors


if __name__ == "__main__":

    print("You only need to download the data once!")
    supported_datasets = ["cmumosei", "cmumosi", "pom"]
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--dataset_name', required=True,
                        type=str, nargs='+', help='Dataset name from possible (cmumosei, cmumosi and pom)')
    args = parser.parse_args()

    dataset_name = args.dataset_name[0]
    if dataset_name not in supported_datasets:
        raise ValueError("Unsupported dataset. Only supported datasets are cmumosei, cmumosi and pom")

    _ = get_and_process_data(dataset_name)
