from ecml_tools.data import open_dataset

# ds = open_dataset(
#     {
#             "dataset": "/home/mlx/ai-ml/datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v4.zarr",
#             "frequency": "6h",
#             "select": ["z_500", "t_850"],
#         }
# )
ds = open_dataset(
    [
        {
            "dataset": "/home/mlx/ai-ml/datasets/experimental/aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v4-ml.zarr",
        },
        {
            "dataset": "/home/mlx/ai-ml/datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v4.zarr",
            "frequency": "6h",
            "select": ["z_500", "t_850"],
        },
    ],
    reorder=[
        "z_500",
        "t_850",
        "q_48",
        "q_56",
        "q_60",
        "q_65",
        "q_68",
        "q_73",
        "q_74",
        "q_79",
        "q_81",
        "q_83",
        "q_90",
        "q_96",
        "q_101",
        "q_105",
        "q_114",
        "q_120",
        "q_133",
        "q_137",
        "t_48",
        "t_56",
        "t_60",
        "t_65",
        "t_68",
        "t_73",
        "t_74",
        "t_79",
        "t_81",
        "t_83",
        "t_90",
        "t_96",
        "t_101",
        "t_105",
        "t_114",
        "t_120",
        "t_133",
        "t_137",
        "u_48",
        "u_56",
        "u_60",
        "u_65",
        "u_68",
        "u_73",
        "u_74",
        "u_79",
        "u_81",
        "u_83",
        "u_90",
        "u_96",
        "u_101",
        "u_105",
        "u_114",
        "u_120",
        "u_133",
        "u_137",
        "v_48",
        "v_56",
        "v_60",
        "v_65",
        "v_68",
        "v_73",
        "v_74",
        "v_79",
        "v_81",
        "v_83",
        "v_90",
        "v_96",
        "v_101",
        "v_105",
        "v_114",
        "v_120",
        "v_133",
        "v_137",
        "w_48",
        "w_56",
        "w_60",
        "w_65",
        "w_68",
        "w_73",
        "w_74",
        "w_79",
        "w_81",
        "w_83",
        "w_90",
        "w_96",
        "w_101",
        "w_105",
        "w_114",
        "w_120",
        "w_133",
        "w_137",
        "vo_48",
        "vo_56",
        "vo_60",
        "vo_65",
        "vo_68",
        "vo_73",
        "vo_74",
        "vo_79",
        "vo_81",
        "vo_83",
        "vo_90",
        "vo_96",
        "vo_101",
        "vo_105",
        "vo_114",
        "vo_120",
        "vo_133",
        "vo_137",
        "d_48",
        "d_56",
        "d_60",
        "d_65",
        "d_68",
        "d_73",
        "d_74",
        "d_79",
        "d_81",
        "d_83",
        "d_90",
        "d_96",
        "d_101",
        "d_105",
        "d_114",
        "d_120",
        "d_133",
        "d_137",
        "z",
        "sp",
        "msl",
        "lsm",
        "sdor",
        "slor",
        "10u",
        "10v",
        "2t",
        "2d",
        "skt",
        "sd",
        "tcw",
        "cp",
        "tp",
        "cos_latitude",
        "cos_longitude",
        "sin_latitude",
        "sin_longitude",
        "cos_julian_day",
        "cos_local_time",
        "sin_julian_day",
        "sin_local_time",
        "insolation",
    ],
)
print(ds.variables)


ds.source(0).dump()
