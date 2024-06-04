# evaluate
import pandas as pd
from scoring_program.evaluation import Metrics

def evaluate(df):
    results = Metrics.evaluate_property_wise_json_based(
        label_list=list(df["ground_truth"]), prediction_list=list(df["annotation"])
    )

    results.update(
        Metrics.evaluate_rouge(label_list=list(df["ground_truth"]), prediction_list=list(df["annotation"]))
    )
    return pd.DataFrame([results])