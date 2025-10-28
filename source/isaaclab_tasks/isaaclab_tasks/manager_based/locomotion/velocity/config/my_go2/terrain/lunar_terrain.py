import isaaclab.terrains as terrain_gen

##
# Scene definition
##
LUNAR_DUMMY_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=1.6,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "repeated": terrain_gen.MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.1,
        #     platform_width=0.5,
        #     max_height_noise=0.2,
        #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=20,
        #         height=0.1,
        #         size=(0.3, 0.5),
        #         max_yx_angle=10.0,
        #         degrees=True
        #     ),
        #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=50,
        #         height=0.2,
        #         size=(0.5, 0.5),
        #         max_yx_angle=30.0,
        #         degrees=True
        #     )
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.03, 0.03), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.34), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.34), platform_width=1.0, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.03, 0.3), num_waves=2, border_width=0.1
        ),
    },
)