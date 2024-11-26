from dataclasses import dataclass, field

from astropy.table import Table
from astropy import units
import pandas as pd
import numpy as np
import warnings

@dataclass
class ClimateData:
    all_data : Table
    location : str = ""
    extreme_type : str = ""
    extreme_low : bool = False
    unit_system : str = "metric"
    celsius : bool = True

    def __post_init__(self):
        self._locations = None
        self._selection = None
        self._selected_data = []
        self._unit_temp = units.deg_C

        year = [w.split("-")[0] for w in self.all_data['DATE']]
        month = [w.split("-")[1] for w in self.all_data['DATE']]
        day = [w.split("-")[2] for w in self.all_data['DATE']]
        self.all_data['YEAR'] = np.array(year, dtype=np.int32)
        self.all_data['MONTH'] = month
        self.all_data['DAY'] = day

        self.all_data['TMAX'].unit = self._unit_temp
        self.all_data['TMIN'].unit = self._unit_temp

    @property
    def locations(self):
        if not self._locations:
            self._locations = sorted(set(self.all_data['NAME']))
        return self._locations

    def set_location(self, location):
        self.location = location
        self._selection = (self.all_data['NAME'] == self.location)
        self._selected_data = self.all_data[self._selection]

    @property
    def selected_data(self):
        return self._selected_data

    def set_unit_system(self, system):
        if system.lower() not in ["metric", "english"]:
            raise ValueError("Valid systems are 'metric' or 'english'")

        if system.lower() == "english":
            pass

    @property
    def years(self):
        use = self._selected_data if self._selected_data else self.all_data
        return use['YEAR'].astype(np.int32)

    def data_year_filter(self, min_year, max_year):
        use = self._selected_data if self._selected_data else self.all_data
        keepers = (use['YEAR'] >= min_year) & (use['YEAR'] <= max_year)
        return self._selected_data[keepers]


@dataclass
class TempData:
    """
    Class to hold temperature data

    Parameters
    ----------
    data : Table
        Table of data that includes a temperature column and a year column
    display_unit : astropy.units.Unit
        The unit to display the temperature in
    extreme_type : str
        The type of extreme to look for.  Valid values are 'high' and 'low'
    """
    data : Table = field(default_factory=Table)
    display_unit : units = units.deg_C
    extreme_type : str = "high"
    start_year : int = 1900
    end_year : int = 2023
    _colnames = dict(high="TMAX", low="TMIN")

    def __post_init__(self):
        self.start_year = self.data['YEAR'].astype(np.int32).min()
        self.end_year = self.data['YEAR'].astype(np.int32).max()

    @property
    def display_temp(self):
        """
        Return the temperature in the display unit
        """
        years_selector = (self.data['YEAR'] >= self.start_year) & (self.data['YEAR'] <= self.end_year)
        return (self.data[self._colnames[self.extreme_type]][years_selector]
                .to(self.display_unit, equivalencies=units.temperature()))

    @property
    def data_year_filter(self):
        """
        Return the data filtered by year
        """
        years_selector = (self.data['YEAR'] >= self.start_year) & (self.data['YEAR'] <= self.end_year)
        return self.data[years_selector]

    @property
    def year_range(self):
        """
        Return the year range
        """
        return (self.data['YEAR'].astype(np.int32).min(),
                self.data['YEAR'].astype(np.int32).max())

    def fit_trend(self, group_by=None, aggregate_by=np.nanmean):

        try:
            badd = self.display_temp.mask
        except AttributeError:
            badd = np.zeros_like(self.display_temp.value, dtype=bool)

        foo = self.data_year_filter[~badd]
        foo[self._colnames[self.extreme_type]] = foo[self._colnames[self.extreme_type]].filled(np.nan)

        if group_by is not None:
            foo = foo.group_by(group_by)
            with warnings.catch_warnings(action="ignore"):
                new_foo = foo.groups.aggregate(aggregate_by)
        else:
            new_foo = foo

        column = self._colnames[self.extreme_type]

        good_temps = ~np.isnan(new_foo[column])
        coeff = np.polyfit(new_foo["YEAR"][good_temps], new_foo[column][good_temps]
                           .to(self.display_unit, equivalencies=units.temperature()), 1)
        p = np.polynomial.Polynomial(coeff[::-1])

        return p

    def max_min_temps_by_year(self, group_by=None, aggregate_by=np.nanmean):
        """
        Return the max or min temperature by year

        Parameters
        ----------
        group_by : str, optional
            The column to group by.  If None, then no grouping is done.
        """
        if group_by is not None:
            # Remove any masked temperature rows
            mask = np.zeros(len(self.data), dtype=bool)
            for column in self._colnames.values():
                mask = mask | self.data[column].mask

            foo = self.data[~mask]
            foo = foo.group_by(group_by)
            with warnings.catch_warnings(action="ignore"):
                new_foo = foo.groups.aggregate(aggregate_by)
        else:
            new_foo = self.data

        column = self._colnames[self.extreme_type]
        return new_foo['YEAR'], new_foo[column].to(self.display_unit, equivalencies=units.temperature())

    def to_pandas_series(self, no_units=True):
        """
        Return a pandas series of the data

        Parameters
        ----------
        no_units : bool, optional
            If True, then the units are removed from the data.
        """
        return pd.Series(data=self.display_temp.value, index=pd.to_datetime(self.data_year_filter['DATE']))