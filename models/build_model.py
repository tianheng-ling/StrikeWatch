from fp32 import Float1DCNN, Float1DSepCNN, FloatRNN, FloatTransformer
from quant import Quant1DCNN, Quant1DSepCNN, QuantRNN, QuantTransformer


def build_model(model_config: dict, enable_qat: bool) -> object:

    model_type = model_config.get("model_type")
    if enable_qat:
        model_map = {
            "cnn": Quant1DCNN,
            "sepcnn": Quant1DSepCNN,
            "rnn": QuantRNN,
            "transformer": QuantTransformer,
        }
    else:
        model_map = {
            "cnn": Float1DCNN,
            "sepcnn": Float1DSepCNN,
            "rnn": FloatRNN,
            "transformer": FloatTransformer,
        }
    return model_map[model_type](**model_config)
