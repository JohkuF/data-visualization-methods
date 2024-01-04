from functools import wraps
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def validate_num_of_images(func):
    """
    A decorator that checks whether the num_of_images argument exceeds the number of columns in the data.

    Args:
        func: The function to be decorated.
git@gitlab.paivola.fi:joalanen/elisa-projekti.git
    Returns:
        A decorated function that checks the num_of_images argument.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function that checks the num_of_images argument.

        Args:
            *args: The positional arguments of the decorated function.
            **kwargs: The keyword arguments of the decorated function.

        Returns:
            The result of the decorated function.

        Raises:
            ValueError: If the num_of_images argument exceeds the number of columns in the data.
        """
        data = args[1] if len(args) > 1 else kwargs.get("data")
        num_of_images = args[2] if len(args) > 1 else kwargs.get("num_of_images")

        if num_of_images is not None and num_of_images > len(data.columns):
            raise ValueError(
                "num_of_images should not exceed the number of columns in the data"
            )
        return func(*args, **kwargs)

    return wrapper


def validate_add_line(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        position = args[1] if len(args) > 1 else kwargs.get("position")
        axis = args[0] if len(args) > 2 else kwargs.get("axis")
        color = args[2] if len(args) > 3 else kwargs.get("color")

        # -----Determine type of input-----

        # Multiple lines need to be drawn
        if isinstance(position, list):
            if not isinstance(axis, list) or not isinstance(color, list):
                raise ValueError(
                    "If position is not list, axis or color can't be a list"
                )

            lines = len(position)

            if not (len(axis) == lines) or not (len(color) == lines):
                raise ValueError(
                    "Axis and color needs to be the same length as position"
                )

            kwargs["num_of_lines"] = lines

        # Only one line needs to be drawn
        else:
            if isinstance(axis, list) or isinstance(color, list):
                raise ValueError(
                    "If position is a not list, axis or color can't be a list"
                )
            kwargs["num_of_lines"] = 1
            pass

        return func(*args, **kwargs)

    return wrapper


def make_subplots(func):
    """
    A decorator that creates a figure and subplots grid, and passes them as keyword arguments to the wrapped function.

    Args:
        func: The function to be wrapped.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Creates a figure and subplots grid, and passes them as keyword arguments to the wrapped function.

        Args:
            *args: The positional arguments to be passed to the wrapped function.
            **kwargs: The keyword arguments to be passed to the wrapped function.

        Returns:
            The result of the wrapped function call.
        """

        # Extract the values of the `num_of_images`, `num_rows`, and `custom_fig_size` parameters
        # from either the positional arguments or the keyword arguments
        num_of_images: int = args[2] if len(args) > 1 else kwargs.get("num_of_images")
        num_rows: int = args[4] if len(args) > 4 else kwargs.get("num_rows", 4)
        custom_fig_size = args[5] if len(args) > 5 else kwargs.get("custom_fig_size")

        # Determine the number of columns based on the number of images and number of rows
        num_columns = int(np.ceil(num_of_images / num_rows))

        # Setting a figsize
        if custom_fig_size:
            figsize = custom_fig_size
        else:
            figsize = (num_columns * 10, num_rows * 14)

        # Create the figure and subplots grid
        fig, ax = plt.subplots(num_columns, num_rows, figsize=figsize)
        ax = ax.flatten()

        # Pass the `fig` and `ax` arguments to the wrapped function
        kwargs["fig"] = fig
        kwargs["ax"] = ax

        # Call the wrapped function with the modified arguments
        return func(*args, **kwargs)

    return wrapper
