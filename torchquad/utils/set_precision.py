import torch

import logging

logger = logging.getLogger(__name__)


def set_precision(data_type="float"):
    """This function allows the user to set the default precision. Call before declaring your variables.

    Args:
        data_type (string, optional): Data type to use, either "float", "double", "complex32", or "complex64". Defaults to "complex32".

    """

    if torch.cuda.is_initialized():
        if data_type.lower() == "double":
            logging.info(
                "Setting default tensor type to cuda.Float64 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif data_type.lower() == "float":
            logging.info(
                "Setting default tensor type to cuda.Float32 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif data_type.lower() == "complex64":
            logging.info(
                "Setting default tensor type to cuda.complex64 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.complex64)
        elif data_type.lower() == "complex32":
            logging.info(
                "Setting default tensor type to cuda.complex32 (CUDA is initialized)."
            )
            torch.set_default_tensor_type(torch.cuda.complex32)
        else:
            logging.error(
                data_type,
                "Invalid data type. Only float or double are supported. "
                "Setting default tensor type to cuda.Float32 (CUDA is initialized). "
                "See 'Message' in the line above for the data type used.",
            )
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        if data_type.lower() == "double":
            logging.info(
                "Setting default tensor type to Float64 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif data_type.lower() == "float":
            logging.info(
                "Setting default tensor type to Float32 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.FloatTensor)
        elif data_type.lower() == "complex64":
            logging.info(
                "Setting default tensor type to complex64 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif data_type.lower() == "complex32":
            logging.info(
                "Setting default tensor type to complex32 (CUDA not initialized)."
            )
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            logging.error(
                data_type,
                "Invalid data type. Only float or double are supported. "
                "Setting default tensor type to Float32. "
                "See 'Message' in the line above for the data type used.",
            )
            torch.set_default_tensor_type(torch.FloatTensor)
