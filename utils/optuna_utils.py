import json


def save_trials_records(json_path: str, study: object, only_best: bool):
    "currently only support single-objective and two-objective search"

    trials_data = []
    trials = study.best_trials if only_best else study.trials
    for t in trials:
        objective1 = None
        objective2 = None

        if t.values is not None:
            if isinstance(t.values, (list, tuple)):
                objective1 = (
                    round(t.values[0], 3)
                    if len(t.values) > 0 and t.values[0] is not None
                    else None
                )
                objective2 = (
                    round(t.values[1], 3)
                    if len(t.values) > 1 and t.values[1] is not None
                    else None
                )
            else:  # single-objective: float
                objective1 = round(t.values, 3)

        formatted_params = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in t.params.items()
        }

        trial_record = {
            "trial": t.number,
            "objective1": objective1,
            "objective2": objective2,  # will be None for single-objective search
            "params": formatted_params,
            "user_attrs": t.user_attrs,
            "state": t.state.name,
        }
        trials_data.append(trial_record)

    with open(json_path, "w") as f:
        json.dump(trials_data, f, indent=4)
