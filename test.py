from ecml_tools.data import open_dataset

name = "aifs-ea-an-oper-0001-mars-o96-2021-6h-v2-only-z"
# # aifs-od-an-oper-0001-mars-o96-2021-6h-v2"
# z = open_dataset(name)
# z.save("new-zarr.zarr")
# exit(0)


# print(z.shape)
# print(z.dates)

z = open_dataset(
    "aifs-ea-an-oper-0001-mars-o96-2021-6h-v2-only-z",
    "aifs-ea-an-oper-0001-mars-o96-2021-6h-v2-without-z",
    end="2021-01-01",
)
print(len(z))

# print(z)

z.save("new-zarr.zarr")
exit(0)

z = open_dataset(name, frequency="1d")
# print(z.shape)
# print(z.dates)

z = open_dataset(name, frequency="1d", start=2021, end=2022)


z = open_dataset(
    "aifs-ea-an-oper-0001-mars-o96-2020-6h-v2",
    "aifs-ea-an-oper-0001-mars-o96-2021-6h-v2",
)

print(z)
print(len(z))

z = open_dataset(
    "aifs-ea-an-oper-0001-mars-o96-2021-6h-v2-only-z",
    "aifs-ea-an-oper-0001-mars-o96-2021-6h-v2-without-z",
)

# z.save('new-zarr.zarr')

print(z)
print(z.shape)
print(len(z))

for i, e in enumerate(z):
    print(i, e)
    if i > 10:
        break

exit()

train = open_dataset(
    "aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v2",
    begin=1979,
    end=2000,
    frequency="6h",
)

test = open_dataset(
    "aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v2",
    begin=2001,
    end=2022,
    frequency="6h",
)


test.sel(time=slice("2001-01-01", "2001-01-02"))
