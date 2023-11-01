from parselmouth.praat import run_file
import onnxruntime as rt
import pandas as pd
import numpy as np
import os
from glob import glob
import logging


def run_myspsolution_praat_file(audio_file: str):
    praat_source_file = "praat/myspsolution.praat"
    audio_file_dir = os.path.dirname(audio_file)

    assert os.path.isfile(audio_file), "Audio file not found"
    assert os.path.isfile(praat_source_file), "Incorrect path to praat script"

    audio_file = os.path.abspath(audio_file)

    objects = run_file(
        praat_source_file,
        -20,
        2,
        0.3,
        "yes",
        audio_file,
        audio_file_dir,
        80,
        400,
        0.01,
        capture_output=True,
    )

    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    logging.debug("parselmouth.Sound response", objects[0])

    result = dict(
        zip(
            [
                "number_of_syllables",
                "number_of_pauses",
                "rate_of_speech",
                "articulation_rate",
                "speaking_duration",
                "original_duration",
                "balance",
                "f0_mean",
                "f0_std",
                "f0_median",
                "f0_min",
                "f0_max",
                "f0_quantile25",
                "f0_quan75",
            ],
            str(objects[1]).strip().split(),
        )
    )

    # Convert strings to numbers
    result["number_of_syllables"] = int(result["number_of_syllables"])
    result["number_of_pauses"] = int(result["number_of_pauses"])
    result["rate_of_speech"] = float(result["rate_of_speech"])
    result["articulation_rate"] = float(result["articulation_rate"])
    result["speaking_duration"] = float(result["speaking_duration"])
    result["original_duration"] = float(result["original_duration"])
    result["balance"] = float(result["balance"])
    result["f0_mean"] = float(result["f0_mean"])
    result["f0_std"] = float(result["f0_std"])
    result["f0_median"] = float(result["f0_median"])
    result["f0_min"] = float(result["f0_min"])
    result["f0_max"] = float(result["f0_max"])
    result["f0_quantile25"] = float(result["f0_quantile25"])
    result["f0_quan75"] = float(result["f0_quan75"])

    return result


def run_mltrnl_praat_file(audio_file: str):
    praat_source_file = "praat/MLTRNL.praat"
    audio_file_dir = os.path.dirname(audio_file)

    assert os.path.isfile(audio_file), "Audio file not found"
    assert os.path.isfile(praat_source_file), "Incorrect path to praat script"

    audio_file = os.path.abspath(audio_file)

    objects = run_file(
        praat_source_file,
        -20,
        2,
        0.3,
        "yes",
        audio_file,
        audio_file_dir,
        80,
        400,
        0.01,
        capture_output=True,
    )

    result = dict(
        zip(
            [
                "avepauseduratin",
                "avelongpause",
                "speakingtot",
                "avenumberofwords",
                "articulationrate",
                "inpro",
                "f1norm",
                "mr",
                "q25",
                "q50",
                "q75",
                "std",
                "fmax",
                "fmin",
                "vowelinx1",
                "vowelinx2",
                "formantmean",
                "formantstd",
                "nuofwrds",
                "npause",
                "ins",
                "fillerratio",
                "xx",
                "xxx",
                "totsco",
                "xxban",
                "speakingrate",
            ],
            str(objects[1]).strip().split(),
        )
    )

    result["avepauseduratin"] = float(result["avepauseduratin"])
    result["avelongpause"] = float(result["avelongpause"])
    result["speakingtot"] = float(result["speakingtot"])
    result["avenumberofwords"] = float(result["avenumberofwords"])
    result["articulationrate"] = float(result["articulationrate"])
    result["inpro"] = float(result["inpro"])
    result["f1norm"] = float(result["f1norm"])
    result["mr"] = float(result["mr"])
    result["q25"] = float(result["q25"])
    result["q50"] = float(result["q50"])
    result["q75"] = float(result["q75"])
    result["std"] = float(result["std"])
    result["fmax"] = float(result["fmax"])
    result["fmin"] = float(result["fmin"])
    result["vowelinx1"] = float(result["vowelinx1"])
    result["vowelinx2"] = float(result["vowelinx2"])
    result["formantmean"] = float(result["formantmean"])
    result["formantstd"] = float(result["formantstd"])
    result["nuofwrds"] = int(result["nuofwrds"])
    result["npause"] = int(result["npause"])
    result["ins"] = int(result["ins"])
    result["fillerratio"] = float(result["fillerratio"])
    result["xx"] = int(result["xx"])
    result["xxx"] = str(result["xxx"])
    result["totsco"] = float(result["totsco"])
    result["xxban"] = str(result["xxban"])
    result["speakingrate"] = float(result["speakingrate"])

    return result


def run_onnx_model(model_name: str, input_data: pd.DataFrame):
    sess = rt.InferenceSession(model_name, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: input_data.astype(np.float32)})[0]
    return pred_onx


def score_cefr_level(audio_file: str):
    res = run_mltrnl_praat_file(audio_file)

    df = pd.DataFrame(res, index=[0])

    scoring_df = df[
        [
            "avepauseduratin",
            "avelongpause",
            "speakingtot",
            "articulationrate",
            "mr",
            "q50",
            "std",
            "fmax",
            "fmin",
            "vowelinx2",
            "formantmean",
            "formantstd",
            "ins",
        ]
    ]

    results = {}

    models = glob("models/onnx/*.onnx")

    for model in models:
        model_name = os.path.basename(model)
        pred_onx = run_onnx_model(model, scoring_df)
        results[model_name] = pred_onx[0]

    return results
