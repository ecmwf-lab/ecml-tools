# ecml-tools

A package to hold various functions to support training of ML models on ECMWF data.

# Datasets

A `dataset` wraps a `zarr` file that follows the format used by ECMWF to train its machine learning models.

```python
from ecml_tools.data import open_dataset

ds = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2")


```

The dataset can be passed as a path or URL to a `zarr` file, or as a name. In the later case, the package will use the entry `zarr_root` of `~/.ecml-tool` file to create the full path or URL.

## Subsetting datasets

You can create a view on the `zarr` file that selects a subset of dates.

### Changing the frequency
```python
from ecml_tools.data import open_dataset

ds = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
    freqency='12h')


```

The `frequency` parameter can be a integer (in hours) or a string following with the suffix `h` (hours) or `d` (days).

### Selecting years
You can select ranges of years using the `start` and `end` keywords:
```python
from ecml_tools.data import open_dataset

training = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
    start=1979,
    end=2020)

test = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2"
    start=2021,
    end=2022)

```

### Combining both

You can combine both subsetting methods:
```python
from ecml_tools.data import open_dataset

training = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
    start=1979,
    end=2020,
    frequency='6h')
