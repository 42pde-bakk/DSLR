import numpy as np


def accepts(*types):
    def check_accepts(f):
        if len(types) != f.__code__.co_argcount:
            return None

        def new_f(*args, **kwargs):
            if any(not isinstance(arg, t) for arg, t in zip(args, types)):
                return None
            return f(*args, **kwargs)

        # new_f.__name__ = f.__name__
        return new_f

    return check_accepts


# noinspection PyTypeChecker
def true_positives(y: np.ndarray, y_hat: np.ndarray, pos_label: int = 1) -> int:
    return np.sum((y == pos_label) & (y_hat == pos_label))


# noinspection PyTypeChecker
def true_negatives(y: np.ndarray, y_hat: np.ndarray, pos_label: int = 1) -> int:
    return np.sum((y != pos_label) & (y_hat != pos_label))


# noinspection PyTypeChecker
def false_positives(y: np.ndarray, y_hat: np.ndarray, pos_label: int = 1) -> int:
    return np.sum((y != pos_label) & (y_hat == pos_label))


# noinspection PyTypeChecker
def false_negatives(y: np.ndarray, y_hat: np.ndarray, pos_label: int = 1) -> int:
    return np.sum((y == pos_label) & (y_hat != pos_label))


@accepts(np.ndarray, np.ndarray)
def accuracy_score_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    return np.sum(y == y_hat) / y.shape[0]


@accepts(np.ndarray, np.ndarray, (int, str))
def precision_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1) -> float:
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    tp = true_positives(y, y_hat, pos_label)
    fp = false_positives(y, y_hat, pos_label)
    return tp / (tp + fp)


@accepts(np.ndarray, np.ndarray, (int, str))
def recall_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1) -> float:
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    tp = true_positives(y, y_hat, pos_label)
    fn = false_negatives(y, y_hat, pos_label)
    return tp / (tp + fn)


@accepts(np.ndarray, np.ndarray, (int, str))
def f1_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1) -> float:
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    tp = true_positives(y, y_hat, pos_label)
    fp = false_positives(y, y_hat, pos_label)
    fn = false_negatives(y, y_hat, pos_label)
    tn = true_negatives(y, y_hat, pos_label)
    print(f'{tp=},{fp=},{fn=},{tn=}')
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return (2 * precision * recall) / (precision + recall)
