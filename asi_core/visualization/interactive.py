import numpy as np
import pandas as pd
import hvplot.pandas
import panel as pn
from panel.io.save import save  as pn_save

pn.extension()  # Enable Panel for interactive visualization

class InteractiveTimeSeriesPlot:
    """
    Interactive time series plot with date and variable selection.

    This class generates an interactive time series visualization using a pandas DataFrame
    with a DatetimeIndex. It allows users to select specific dates and variables (columns)
    to visualize through widgets like MultiSelect and DatePicker. Navigation buttons
    enable browsing between available dates.

    Attributes:
        df (pd.DataFrame): Input time series DataFrame.
        plot_kwargs (dict): Additional keyword arguments for `hvplot.line()`.
        fontsize (dict): Font sizes for plot elements.
        available_dates (list): Sorted list of unique dates from the DataFrame index.
        columns (list): List of column names in the DataFrame.
        variable_selector (pn.widgets.MultiSelect): Widget for selecting variables to plot.
        date_picker (pn.widgets.DatePicker): Widget for selecting a date.
        prev_button (pn.widgets.Button): Button for selecting the previous date.
        next_button (pn.widgets.Button): Button for selecting the next date.
        plot_pane (pn.pane.HoloViews): Pane for displaying the plot.
        layout (pn.Column): The overall Panel layout for the interactive plot.

    Parameters:
        df (pd.DataFrame): A time series DataFrame with a DatetimeIndex.
        **plot_kwargs: Additional keyword arguments for `hvplot.line()` used for plotting customization.

    Raises:
        ValueError: If `df` is not a non-empty DataFrame with a DatetimeIndex.
        ValueError: If no valid dates are found in `df`.

    Example usage:
        >>> plot = InteractiveTimeSeriesPlot(my_dataframe, title="My Time Series: ")
        >>> plot.show()
        >>> plot.save_as_html("output_plot.html")
    """
    def __init__(self, df, **plot_kwargs):
        """
        Initializes an interactive time series plot.

        Parameters:
            df (pd.DataFrame): A time series DataFrame with a DatetimeIndex.
            **plot_kwargs: Additional keyword arguments for hvplot.line().
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame with a DatetimeIndex.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex.")

        self.df = df
        self.plot_kwargs = plot_kwargs
        self.fontsize = plot_kwargs.pop('fontsize',
                                        {'title': 18, 'xlabel': 14, 'ylabel': 14, 'xticks': 12, 'yticks': 12})
        self.available_dates = sorted(np.unique(df.index.date))  # Unique available dates
        self.columns = list(df.columns)  # List of available columns (time series variables)

        if not self.available_dates:
            raise ValueError("No valid dates found in dataset.")

        # Dropdown MultiSelect widget for variable selection
        self.variable_selector = pn.widgets.MultiSelect(
            name="Select Variables",
            options=self.columns,
            value=self.columns,  # Default: all selected
            size=min(len(self.columns), 6)  # Dynamic height
        )

        # Panel widgets with restricted date selection
        self.date_picker = pn.widgets.DatePicker(
            name="Select Date",
            value=self.available_dates[0],
            options=self.available_dates  # Restrict selectable dates
        )

        self.prev_button = pn.widgets.Button(name="Previous Day", button_type="primary")
        self.next_button = pn.widgets.Button(name="Next Day", button_type="primary")

        # Create the plot pane
        self.plot_pane = pn.pane.HoloViews(self.plot_day(self.date_picker.value, self.variable_selector.value))

        # Link events
        self.prev_button.on_click(self.prev_date)
        self.next_button.on_click(self.next_date)
        self.date_picker.param.watch(self.update_plot, "value")
        self.variable_selector.param.watch(self.update_plot, "value")

        # Create the layout
        self.layout = pn.Column(
            pn.Row(self.prev_button, self.date_picker, self.next_button),
            self.variable_selector,  # Dropdown list for variable selection
            self.plot_pane,
        )

    def plot_day(self, selected_date, selected_variables):
        """
        Filters and plots the data for the selected date and variables.

        :param selected_date: The selected date for plotting.
        :type selected_date: datetime.date
        :param selected_variables: Selected columns to plot.
        :type selected_variables: list
        :returns: hvplot object
        """
        selected_date = pd.Timestamp(selected_date)
        kwargs = self.plot_kwargs.copy()
        title = f"{kwargs.pop('title')}{selected_date.date()}"
        daily_data = self.df[self.df.index.date == selected_date.date()]

        if daily_data.empty or not selected_variables:
            return "No data available for this date or no variables selected"

        return daily_data[selected_variables].hvplot.line(title=title, **kwargs).opts(fontsize=self.fontsize)

    def update_plot(self, event):
        """ Updates the plot when the selected date or variables change. """
        self.plot_pane.object = self.plot_day(self.date_picker.value, self.variable_selector.value)

    def prev_date(self, event):
        """ Moves to the previous available date in the dataset. """
        current_index = self.available_dates.index(self.date_picker.value)
        if current_index > 0:
            self.date_picker.value = self.available_dates[current_index - 1]

    def next_date(self, event):
        """ Moves to the next available date in the dataset. """
        current_index = self.available_dates.index(self.date_picker.value)
        if current_index < (len(self.available_dates) - 1):
            self.date_picker.value = self.available_dates[current_index + 1]

    def save_as_html(self, filename="interactive_plot.html"):
        """ Saves the interactive plot as an HTML file. """
        pn_save(self.layout, filename, embed=True)
        print(f"Plot saved as {filename}")

    def show(self):
        """ Displays the Panel layout. """
        return self.layout
