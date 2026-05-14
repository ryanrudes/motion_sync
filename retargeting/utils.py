import numpy as np
import torch


def summarize_nested_dict(nested_dict: dict, indent: int = 0, indent_str: str = "  ") -> str:
    result = ""
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            result += f"{indent_str * indent}{key}:\n"
            result += summarize_nested_dict(value, indent + 1, indent_str)
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            result += f"{indent_str * indent}{key}: {value.shape} {value.dtype}\n"
        else:
            result += f"{indent_str * indent}{key}: {value}\n"
    return result