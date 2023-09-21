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
    freqency="12h")


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
    frequency="6h")
```


## Combining datasets

You can create a virtual dataset by combining two or more `zarr` files.

```python
from ecml_tools.data import open_dataset

ds = open_dataset(
    "dataset-1",
    "dataset-2",
    "dataset-3",
    ...
)

```


When given a list of `zarr` files, the package will automatically work out if the files can be _concatenated_ or _joined_ by looking at the range of dates covered by each files.


## Concatenating datasets

You can concatenate two or more datasets along the dates dimension. The package will check that all datasets are compatible (same resolution, same variables, etc.). Currently, the datasets must be given in chronological order with no gaps between them.

```python
from ecml_tools.data import open_dataset

ds = open_dataset(
    "aifs-ea-an-oper-0001-mars-o96-1940-1978-1h-v2",
    "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2"
)

```

Please note that you can pass more than two `zarr` files to the function.

## Joining datasets

You can join two datasets that have the same dates, combining their variables.

```python
from ecml_tools.data import open_dataset

ds = open_dataset(
    "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
    "some-extra-parameters-from-another-source-o96-1979-2022-1h-v2",
)

```

|a|b|c|d| + |e|f|g|h| => |a|b|c|d|e|f|g|h|

If a variable is present in more that one file, that last occurrence of that variable will be used, and will be position of the first occurrence of that name.

Please note that you can join more than two `zarr` files.
