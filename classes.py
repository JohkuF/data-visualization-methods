"""
Containing all the necessary classes used in this project.
"""

from typing import Union, Literal
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from decorators import (
    make_subplots,
    validate_add_line,
    validate_num_of_images,
)


class DataProcess:
    """
    A class for data processing methods.

    This class provides methods for processing data in pandas DataFrame format.

    Methods:
    --------

    unix_timestamp_conventer(
        input_df: pd.DataFrame,
        input_col_name: str,
        output_col_name: str,
        output_col_index: int,
        format: str = "%Y-%m-%d %H:%M:%S",
        should_return: bool = True,
    ) -> pd.DataFrame | None:
        Decodes unix_timestamp to redable format.
    """

    @staticmethod
    def unix_timestamp_conventer(
        input_df: pd.DataFrame,
        input_col_name: str,
        output_col_name: str,
        output_col_index: int,
        output_format: str = "%Y-%m-%d %H:%M:%S",
        should_return: bool = True,
    ) -> pd.DataFrame | None:
        """
        Converts a Unix timestamp column in a pandas DataFrame to a datetime column in the specified format.

        Args:
            input_df: The input DataFrame to convert.
            input_col_name: The name of the column containing the Unix timestamps.
            output_col_name: The desired name of the new datetime column.
            output_col_index: The position in which to insert the new datetime column.
            format: The desired format for the new datetime column (default is "%Y-%m-%d %H:%M:%S").
            should_return: If True, returns a new DataFrame with the converted column (default is True).

        Returns:
            If should_return is True, returns a new pandas DataFrame with the converted column.
            Otherwise, modifies the input_df in place and returns None.
        """

        if should_return:
            input_df = input_df.copy()

        # convert the Unix timestamp to datetime format
        input_df[output_col_name] = pd.to_datetime(
            input_df[input_col_name], unit="s"
        ).dt.strftime(output_format)

        # Move to desired position
        formatted_datetime = input_df.pop(output_col_name)
        input_df.insert(output_col_index, output_col_name, formatted_datetime)

        if should_return:
            return input_df
        return None

    @staticmethod
    def analyze_specific_spot(
        df: pd.DataFrame, column_name: str | int, value_index: int, delta: int
    ) -> pd.DataFrame:
        """
        Returns a subset of a pandas DataFrame that contains rows where the
        value in the column specified by `column_name` is within `delta` of the
        value in the row specified by `value_index`.

        Parameters:
        -----------

        df: The pandas DataFrame to analyze.
            type df: pandas.DataFrame

        value_index: The index of the row to use as the center of the analysis.
            type value_index: int

        column_name: The name of the column in `df` to use for the analysis.
            type column_name: str

        delta: The maximum distance between the value in the center row
        and the values in the other rows to include in the output.
            type delta: int

        Returns:
        --------

        A pandas DataFrame that contains the rows from `df` where the
        value in `column_name` is within `delta` of the value in the row
        specified by `value_index`.
            type: pandas.DataFrame
        """

        # Select rows where the value in `column_name` is within `delta` of the
        # value in the row specified by `value_index`.

        print("B", value_index - delta, value_index + delta, end="\n" * 2)

        selected_rows = df[
            df[column_name].between(value_index - delta, value_index + delta)
        ]

        return selected_rows

    @staticmethod
    def order_corr(
        corr_df: pd.DataFrame, by: str, ascending: bool = False
    ) -> pd.DataFrame:
        """
        Orders the columns of a correlation matrix in descending or ascending order based on a specified column.

        Parameters:
        - corr_df (pd.DataFrame): a correlation matrix
        - by (str): the name of the column to sort by
        - ascending (bool): whether to sort in ascending order (default is False)

        Returns:
        - pd.DataFrame: the ordered correlation matrix
        """

        # Select the row corresponding to the 'by' column and transpose the result to sort by rows
        ordered = (
            corr_df.corr()
            .loc[[by], :]
            .transpose()
            .sort_values(by=by, ascending=ascending)
            .fillna(0)
            .transpose()
        )

        return ordered

    @staticmethod
    def linear_regression():
        # TODO Make the method
        pass

    @staticmethod
    def get_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """
        Counts z-score to each datapoint.
        """
        return df.apply(lambda x: (x - x.mean() / x.std()))


class DataVisualization:
    """
    This class contains useful methods for data visualization that can be used in this project or future projects.

    Methods:
    --------
    bold_labels(plot, columns: list, axis: Union[Literal['y'], Literal['x'], Literal[True]] = True) -> None:
        Applies bold formatting to the labels of a given plot.

    plot_scatterplots(data: pd.DataFrame, num_of_images: int, x_value: str, num_rows: int = 4,
        custom_fig_size: tuple | None = None, save_path: str | None = None) -> DataVisualization:
        Plot multiple scatterplots in a single figure.
    """

    def __init__(self, fig: plt.Figure) -> None:
        """
        Initializes the DataVisualization instance.

        Parameters:
        -----------
        fig : plt.Figure
            The figure to be used in the visualizations.
        """
        self.fig = fig

    @validate_add_line
    def add_line(
        self,
        position: int | float | list,
        axis: str | list = "x",
        color: str | list = "r",
        num_of_lines: int = 1,
    ) -> "DataVisualization":
        """
        Add one or more vertical/horizontal line(s) to the plot.

        Parameters:
        -----------
        position : int, float, or list
            The position(s) of the line(s) to be added. If multiple lines are being
            added, pass a list of positions.

        axis : str or list, optional
            The axis on which to add the line(s). Pass "x" to add vertical line(s)
            and "y" to add horizontal line(s). If adding multiple lines, pass a
            list of axes corresponding to each line.

        color : str or list, optional
            The color(s) of the line(s) to be added. Pass a string or list of strings
            representing valid matplotlib colors. If adding multiple lines, pass a
            list of colors corresponding to each line.

        num_of_lines : int, optional
            The number of lines to be added. Default is 1.

        Returns:
        --------
        DataVisualization:
            An instance of DataVisualization that can be used for method chaining.
        """

        # Check if multiple lines are being added
        multiple_lines = True if num_of_lines > 1 else False

        # Add the line(s) to each subplot
        for i in range(num_of_lines):
            for ax in self.fig.axes:
                if axis[i] == "x":
                    ax.axvline(
                        x=position[i] if multiple_lines else position, color=color[i]
                    )
                if axis[i] == "y":
                    ax.axhline(
                        y=position[i] if multiple_lines else position, color=color[i]
                    )

        return DataVisualization(self.fig)

    @staticmethod
    def bold_labels(
        plot: plt.Axes,
        columns: list,
        axis: Union[Literal["y"], Literal["x"], Literal[True]] = True,
    ) -> None:
        """
        Applies bold formatting to the labels of a given plot.

        Parameters:
        -----------
        plot : plt.Axes
            The plot to apply the formatting to.

        columns : list
            A list of column names or labels to apply bold formatting to.

        axis : Union[Literal['y'], Literal['x'], Literal[True]], optional
            Determines which axis to apply the formatting to. By default, it applies the formatting to both axes.

            * If `axis` is 'y', bold formatting is applied to the y-axis labels.
            * If `axis` is 'x', bold formatting is applied to the x-axis labels.
            * If `axis` is True, bold formatting is applied to both the x- and y-axis labels.

        Raises:
        -------
        TypeError:
            If the type of the `axis` parameter is not a Literal['y'], Literal['x'], or Literal[True].

        Returns:
        --------
        None
        """
        if axis == "y" or axis is True:
            labels_y = plot.get_yticklabels()

            new_labels_y = [
                f"$\mathbf{{{label.get_text()}}}$"
                if label.get_text() in columns
                else label.get_text()
                for label in labels_y
            ]

            plot.set_yticklabels(new_labels_y)

        if axis == "x" or axis is True:
            labels_x = plot.get_xticklabels()

            new_labels_x = [
                f"$\mathbf{{{label.get_text()}}}$"
                if label.get_text() in columns
                else label.get_text()
                for label in labels_x
            ]
            plot.set_xticklabels(new_labels_x)

    @staticmethod
    @validate_num_of_images
    @make_subplots
    def plot_xy_plots(
        plot_type,
        data: pd.DataFrame,
        num_of_images: int,
        x_value: str,
        num_rows: int = 4,
        custom_fig_size: tuple | None = None,
        save_path: str | None = None,
        **kwargs,
    ) -> "DataVisualization":
        """
        Plot multiple scatterplots in a single figure.

        Args:
            plot_type (callable): The type of plot to be generated. Can be any callable that
                generates a scatterplot. Examples include `matplotlib.pyplot.scatter` or
                `seaborn.scatterplot`.
            data (pd.DataFrame): The dataset to be plotted.
            num_of_images (int): The number of scatterplots to be generated.
            x_value (str): The name of the x-axis column to be used in all scatterplots.
            num_rows (int): The number of rows to be used in the subplots grid. Default is 4.
            custom_fig_size (tuple, optional): A tuple of (width, height) of the figure. Default
                is None.
            save_path (str, optional): The path to save the plot image. Default is None.

        Raises:
        -------
        ValueError:
            "Number of images to be generated exceeds number of columns in the data"

        Returns:
        --------
        DataVisualization: An instance of DataVisualization that can be used for method chaining.
        """

        fig, ax = kwargs["fig"], kwargs["ax"]

        # Get the column names of the data
        columns = data.columns.to_numpy()

        # Plot each scatterplot in the subplots
        for i in range(num_of_images):
            plot_type(data=data, x=x_value, y=columns[i], ax=ax[i])

        if save_path is not None:
            plt.savefig(save_path)

        # Return an instance of DataVisualization for method chaining
        return DataVisualization(fig)

    @staticmethod
    def plot_corr_barplot(
        corr_df: pd.DataFrame,
        target: str,
        figsize: tuple[int, int] = (60, 10),
    ) -> None:
        """
        Plots correlation barplot sorted
        """
        # Get correlation matrix for TARGET_CELL and sort columns by descending correlation value
        # Sort rows of the correlation matrix based on column order and replace null values with 0
        # Transpose the matrix to sort rows by descending correlation value
        ordered = DataProcess.order_corr(corr_df, target)

        # Plot
        _, ax = plt.subplots(figsize=figsize)

        sns.barplot(data=ordered, ax=ax)
