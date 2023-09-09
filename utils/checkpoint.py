import paddle

from .logger import logger


def load_pretrained_model_from_path(model: paddle.nn.Layer,
                                    path: str,
                                    verbose: bool = True):
    """
    """
    para_state_dict = paddle.load(path)
    load_pretrained_model_from_state_dict(
        model, para_state_dict, verbose=verbose)

def load_pretrained_model_from_state_dict(model: paddle.nn.Layer,
                                          state_dict: dict,
                                          verbose: bool = True):
    """
    """
    model_state_dict = model.state_dict()
    keys = model_state_dict.keys()
    num_params_loaded = 0

    for k in keys:

        if k not in state_dict:
            if verbose:
                logger.warning("{} is not in pretrained model".format(k))
        elif list(state_dict[k].shape) != list(model_state_dict[k].shape):
            if verbose:
                logger.warning(
                    "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                    .format(k, state_dict[k].shape, model_state_dict[k].shape))
        else:
            model_state_dict[k] = state_dict[k]
            num_params_loaded += 1

    model.set_dict(model_state_dict)
    logger.info("There are {}/{} variables loaded into {}.".format(
        num_params_loaded, len(model_state_dict), model.__class__.__name__))