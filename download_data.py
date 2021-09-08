import os
import re

import mmsdk
import numpy as np
from loguru import logger
from mmsdk import mmdatasdk as md
from slp.config.nlp import SPECIAL_TOKENS
from slp.util.system import pickle_dump, pickle_load, safe_mkdirs
from tqdm import tqdm

MOSEI_COVAREP_FACET_GLOVE = {
    "audio": "CMU_MOSEI_COVAREP",
    "text": "CMU_MOSEI_TimestampedWordVectors",
    "raw": "CMU_MOSEI_TimestampedWords",
    "visual": "CMU_MOSEI_Visual_Facet_42",
    "labels": "CMU_MOSEI_Opinion_Labels",
}


def download_mmdata(base_path, dataset):
    safe_mkdirs(base_path)

    try:
        md.mmdataset(dataset.highlevel, base_path)
    except RuntimeError:
        logger.info("High-level features have been downloaded previously.")

    try:
        md.mmdataset(dataset.raw, base_path)
    except RuntimeError:
        logger.info("Raw data have been downloaded previously.")

    try:
        md.mmdataset(dataset.labels, base_path)
    except RuntimeError:
        logger.info("Labels have been downloaded previously.")


def avg_collapse(intervals, features):
    try:
        return np.average(features, axis=0)
    except Exception as e:
        del e

        return features


def deploy(in_dataset, destination):
    deploy_files = {x: x for x in in_dataset.keys()}
    in_dataset.deploy(destination, deploy_files)


def load_modality(base_path, feature_cfg, modality):
    mfile = feature_cfg[modality]
    path = os.path.join(base_path, "{}.csd".format(mfile))
    logger.info("Using {} for {} modality".format(path, modality))
    data = md.mmdataset(path)

    return data


def get_vocabulary(text_dataset):
    all_words = []

    for seg in text_dataset.keys():
        words = text_dataset[seg]["features"][0]

        for w in words:
            wi = w.decode("utf-8")
            all_words.append(wi)

    all_words = list(set(all_words))

    return all_words


def create_word2idx(all_words):
    word2idx, idx = {}, 0

    for w in sorted(all_words):
        if w not in word2idx:
            word2idx[w] = idx
            idx += 1

    for t in SPECIAL_TOKENS:
        word2idx[t.value] = idx
        idx += 1

    return word2idx


def select_dataset(dataset_name):
    if dataset_name == "mosi":
        dataset = md.cmu_mosi
    elif dataset_name == "mosei":
        dataset = md.cmu_mosei
    elif dataset_name == "pom":
        dataset = md.pom
    else:
        raise ValueError("Unsupported dataset. Use [mosei|mosi|pom]")

    return dataset


def patch_missing_metadata(data):
    # Remove need for annoying input that stops execution

    for k in data.computational_sequences.keys():
        data.computational_sequences[k].metadata[
            "dimension names"
        ] = data.computational_sequences[k].metadata.get("dimension names", None)
        data.computational_sequences[k].metadata[
            "computational sequence version"
        ] = data.computational_sequences[k].metadata.get(
            "computational sequence version", None
        )
        data.computational_sequences[k].metadata[
            "dimension namescomputational sequence version"
        ] = None


def load_and_align(
    base_path,
    dataset="mosei",
    feature_cfg=MOSEI_COVAREP_FACET_GLOVE,
    modalities={"audio", "visual", "text"},
    collapse=None,
):
    dataset = select_dataset(dataset)
    download_mmdata(base_path, dataset)
    recipe = {
        f: os.path.join(base_path, "{}.csd".format(f))

        for k, f in feature_cfg.items()

        if k in list(modalities) + ["raw"]
    }
    data = md.mmdataset(recipe)

    patch_missing_metadata(data)

    if collapse is None:
        collapse = [avg_collapse]
    # first we align to words with averaging
    # collapse_function receives a list of functions

    word_align_path = base_path + "_word_aligned"
    safe_mkdirs(word_align_path)

    data.align(feature_cfg["raw"], collapse_functions=collapse)
    data.impute(feature_cfg["raw"])
    deploy(data, word_align_path)
    all_words = get_vocabulary(data[feature_cfg["raw"]])

    word2idx = create_word2idx(all_words)

    label_recipe = {
        feature_cfg["labels"]: os.path.join(
            base_path, "{}.csd".format(feature_cfg["labels"])
        )
    }
    data.add_computational_sequences(label_recipe, destination=None)
    patch_missing_metadata(data)

    data.align(feature_cfg["labels"])
    data.hard_unify()
    align_path = base_path + "_final_aligned"
    safe_mkdirs(align_path)
    deploy(data, align_path)

    return data, word2idx


def load_dataset(
    base_path,
    dataset="mosei",
    feature_cfg=MOSEI_COVAREP_FACET_GLOVE,
    modalities={"audio", "text", "visual"},
    already_segmented=False,
):
    dataset = select_dataset(dataset)
    download_mmdata(base_path, dataset)
    recipe = {
        f: os.path.join(base_path, "{}.csd".format(f))

        for k, f in feature_cfg.items()

        if k in list(modalities) + ["raw"]
    }
    data = md.mmdataset(recipe)

    patch_missing_metadata(data)

    all_words = get_vocabulary(data[feature_cfg["raw"]])

    word2idx = create_word2idx(all_words)

    label_recipe = {
        feature_cfg["labels"]: os.path.join(
            base_path, "{}.csd".format(feature_cfg["labels"])
        )
    }
    data.add_computational_sequences(label_recipe, destination=None)

    patch_missing_metadata(data)

    if not already_segmented:
        data.align(feature_cfg["labels"])
        data.hard_unify()

    return data, word2idx


def remove_pause_tokens(mods, modalities, is_raw_text_feature):
    # Handle speech pause
    mods_nosp = {k: [] for k in modalities}

    for m in modalities:
        for i in range(len(mods[m])):
            if mods["raw"][i] != "sp":
                mods_nosp[m].append(mods[m][i])

    return mods_nosp


def replace_sp_token(mods, is_raw_text_feature):
    mods_nosp = mods

    if is_raw_text_feature:
        for i in range(len(mods["text"])):
            if mods["text"][i] == "sp":
                mods_nosp["text"][i] = SPECIAL_TOKENS.PAUSE.value

    return mods_nosp


def pad_modality_features(
    mods, modalities, max_length, pad_front, pad_back, is_raw_text_feature
):
    if pad_front and pad_back:
        raise ValueError("Only one of pad_front and pad_back should be true.")

    def compute_padding(m, seglen):
        t = []

        for i in range(max_length[m] - seglen):
            if is_raw_text_feature and m == "text":
                t.append(SPECIAL_TOKENS.PAD.value)
            else:
                vshape = mods[m][0].shape
                pad = np.zeros(vshape)
                t.append(pad)

        return t

    for m in modalities:
        if isinstance(mods[m], np.ndarray):
            mods[m] = [x for x in mods[m]]
        seglen = len(mods[m])

        if seglen >= max_length[m]:
            t = []

            for i in range(max_length[m]):
                t.append(mods[m][i])
            mods[m] = t
        else:
            if pad_front:
                padding = compute_padding(m, seglen)
                mods[m] = padding + mods[m]

            if pad_back:
                padding = compute_padding(m, seglen)
                mods[m] = mods[m] + padding

    return mods


def clean_split_dataset(
    data,
    dataset="mosei",
    feature_cfg=MOSEI_COVAREP_FACET_GLOVE,
    modalities={"audio", "text", "visual"},
    remove_pauses=False,
    max_length=-1,
    pad_front=False,
    pad_back=False,
    aligned=True,
):
    dataset = select_dataset(dataset)
    train_split = dataset.standard_folds.standard_train_fold
    dev_split = dataset.standard_folds.standard_valid_fold
    test_split = dataset.standard_folds.standard_test_fold

    train, dev, test = [], [], []
    num_drop = 0

    segments = data[feature_cfg["labels"]].keys()

    if max_length < 0:
        max_length = {
            m: max([len(data[feature_cfg[m]][s]["features"]) for s in segments])

            for m in list(modalities) + ["raw"]
        }
    else:
        max_length = {m: max_length for m in list(modalities) + ["raw"]}

    for segment in tqdm(segments):
        # get the video ID and the features out of the aligned dataset
        sidx = segment.find("[")

        if sidx > 0:
            vid = segment[:sidx]
        else:
            vid = segment

        # if segment == 'c5xsKMxpXnc':
        mods = {
            k: data[feature_cfg[k]][segment]["features"]

            for k in list(modalities) + ["raw"]
        }

        is_raw_text_feature = isinstance(mods["text"][0][0], bytes)

        raw_text_mods = ["raw", "text"] if is_raw_text_feature else ["raw"]
        mods_with_possible_nan = list(set(modalities) - set(raw_text_mods))
        mods_without_raw_text = list(set(modalities) - set(raw_text_mods))

        for m in mods_with_possible_nan:
            mods[m] = np.nan_to_num(mods[m])

        for m in raw_text_mods:
            words = []

            for i in range(len(mods[m])):
                words.append(mods[m][i][0].decode("utf-8"))
            mods[m] = words

        if aligned:
            num_drop = 0
            # if the sequences are not same length after alignment,
            # there must be some problem with some modalities
            # we should drop it or inspect the data again
            mod_shapes = {k: len(v) for k, v in mods.items()}

            if not len(set(list(mod_shapes.values()))) <= 1:
                logger.warning("Datapoint {} shape mismatch {}".format(vid, mod_shapes))
                num_drop += 1

                continue

            if remove_pauses:
                mods = remove_pause_tokens(mods, modalities, is_raw_text_feature)

        mods = pad_modality_features(
            mods, modalities, max_length, pad_front, pad_back, is_raw_text_feature
        )
        mods = replace_sp_token(mods, is_raw_text_feature)

        for m in mods_without_raw_text:
            mods[m] = np.asarray(mods[m])

        mods["video_id"] = vid
        mods["segment_id"] = segment
        mods["label"] = np.nan_to_num(data[feature_cfg["labels"]][segment]["features"])

        if vid in train_split:
            train.append(mods)
        elif vid in dev_split:
            dev.append(mods)
        elif vid in test_split:
            test.append(mods)
        else:
            logger.warning("{} does not belong to any of the splits".format(vid))
    logger.warning("Dropped {} data points".format(num_drop))

    return train, dev, test


def load_splits(
    base_path,
    dataset="mosei",
    feature_cfg=MOSEI_COVAREP_FACET_GLOVE,
    modalities={"audio", "text", "visual"},
    remove_pauses=False,
    max_length=-1,
    pad_front=False,
    pad_back=False,
    already_aligned=False,
    align_features=True,
    cache=None,
):
    if cache is not None:
        try:
            return pickle_load(cache)
        except FileNotFoundError:
            pass

    if not already_aligned and align_features:
        data, word2idx = load_and_align(
            base_path,
            dataset=dataset,
            feature_cfg=feature_cfg,
            modalities=modalities,
            collapse=[avg_collapse],
        )
    else:
        data, word2idx = load_dataset(
            base_path,
            dataset=dataset,
            feature_cfg=feature_cfg,
            modalities=modalities,
            already_segmented=already_aligned or align_features,
        )

    train, dev, test = clean_split_dataset(
        data,
        dataset=dataset,
        feature_cfg=feature_cfg,
        modalities=modalities,
        remove_pauses=remove_pauses,
        max_length=max_length,
        pad_front=pad_front,
        pad_back=pad_back,
        aligned=already_aligned or align_features,
    )

    if cache is not None:
        pickle_dump((train, dev, test, word2idx), cache)

    return train, dev, test, word2idx


def mosei(
    base_path,
    feature_cfg=MOSEI_COVAREP_FACET_GLOVE,
    modalities={"audio", "text", "visual"},
    remove_pauses=False,
    max_length=-1,
    pad_front=False,
    pad_back=False,
    cache=None,
    already_aligned=False,
    align_features=True,
):
    return load_splits(
        base_path,
        dataset="mosei",
        feature_cfg=feature_cfg,
        modalities=modalities,
        remove_pauses=remove_pauses,
        max_length=max_length,
        pad_front=pad_front,
        pad_back=pad_back,
        cache=cache,
        already_aligned=already_aligned,
        align_features=align_features,
    )


def data_pickle(fname):
    data = pickle_load(fname)

    return data["train"], data["valid"], data["test"], None


if __name__ == "__main__":
    import sys

    base_path = sys.argv[1]
    train, dev, test, w2i = mosei(
        base_path,
        feature_cfg=MOSEI_COVAREP_FACET_GLOVE,
        modalities=["audio", "text", "visual"],
        remove_pauses=True,
        pad_front=False,
        pad_back=False,
        already_aligned=False,
        align_features=True,
    )
