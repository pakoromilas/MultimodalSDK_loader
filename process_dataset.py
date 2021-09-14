import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "CMU-MultimodalSDK"))

from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import log


def deploy(in_dataset,destination):
    deploy_files={x:x for x in in_dataset.keys()}
    in_dataset.deploy(destination,deploy_files)


def download_data(dataset, dataset_name):
    source = {}
    cmumosei_dataset = {}

    if not os.path.isdir(dataset_name + "_raw"):
        source["raw"] = dataset.raw
    else:
        dataset["raw"] = mmdatasdk.mmdataset(dataset_name + "_raw")

    if not os.path.isdir(dataset_name + "_highlevel"):
        source["highlevel"] = dataset.highlevel
    else:
        dataset["highlevel"] = mmdatasdk.mmdataset(dataset_name + "_highlevel")

    if not os.path.isdir(dataset_name + "_labels"):
        source["labels"] = dataset.labels
    else:
        dataset["labels"] = mmdatasdk.mmdataset(dataset_name + "_labels")

    #source = {"raw": dataset.raw, "highlevel": dataset.highlevel, "labels": dataset.labels}

    for key in source:
        cmumosei_dataset[key]=mmdatasdk.mmdataset(source[key], 'cmumosei_%s/'%key)
    return cmumosei_dataset


def process_data(dataset_name, seq_len=50, non_sequences=["All Labels"]):

    folders = [dataset_name + "_highlevel", dataset_name + "_labels"]

    dataset={}
    for folder in folders:
        dataset[folder.split("_")[1]]=mmdatasdk.mmdataset(folder)

    word_aligned_path = dataset_name + '_word_aligned_highlevel'
    if os.path.isdir(word_aligned_path) and os.listdir(word_aligned_path) == []:
        dataset["highlevel"] = mmdatasdk.mmdataset(word_aligned_path)
    else:
        dataset["highlevel"].align("glove_vectors")
        dataset["highlevel"].impute('glove_vectors')
        deploy(dataset["highlevel"], word_aligned_path)

    final_aligned_path = dataset_name + '_final_aligned_highlevel'
    if os.path.isdir(word_aligned_path) and os.listdir(word_aligned_path) == []:
        dataset["highlevel"] = mmdatasdk.mmdataset("final_aligned")
    else:
        dataset["highlevel"].computational_sequences["All Labels"] = dataset["labels"]["All Labels"]

        dataset["highlevel"].align("All Labels")
        dataset["highlevel"].hard_unify()

        deploy(dataset["highlevel"], final_aligned_path)

    tensors = dataset["highlevel"].get_tensors(
        seq_len=seq_len, non_sequences=non_sequences, direction=False,
        folds=[mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
               mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
               mmdatasdk.cmu_mosei.standard_folds.standard_test_fold])

    fold_names = ["train", "valid", "test"]

    for i in range(3):
        for csd in list(dataset["highlevel"].keys()):
            print ("Shape of the %s computational sequence for %s fold is %s"%(
                csd, fold_names[i], tensors[i][csd].shape))

    return tensors


if __name__=="__main__":

    print("You only need to download the data once!")
    dataset_name = "cmumosei"
    dataset = download_data(mmdatasdk.cmu_mosei, dataset_name)
    process_data(dataset_name)
    log.success("Dataset processed")
