import xarray as xr
import synpivimage as spi
from tqdm import tqdm


def _create_synthetic_piv():
    cfg = spi.core.DEFAULT_CFG

    cfg['bit_depth'] = 16
    cfg['nx'] = 128
    cfg['ny'] = 128
    cfg['particle_size_mean'] = 3
    cfg['noise_baseline'] = 100
    cfg['laser_max_intensity'] = 1000

    for _level, ppp in [('low', 0.001), ('medium', 0.1), ('high', 0.6)]:
        intensities = []
        for _ in tqdm(range(4)):
            cfg['particle_number'] = int(ppp*cfg['nx']*cfg['ny'])
            _intensity, _partpos = spi.generate_image(cfg)

            intensities.append(_intensity)

        xr_img = spi.core.combine_particle_image_data_arrays(intensities)
        xr_img.to_netcdf(f'../data/syn_piv_{_level}.nc')


def get_synthetic_piv(density_level='medium') -> xr.DataArray:
    """returns n synthetic images. Three density levels
    are available: 'low', 'medium', 'high'"""
    _level = density_level.lower()
    if _level not in ('low', 'medium', 'high'):
        raise ValueError(f'Level must be "low", "medium" or "high", not "{density_level}"')
    return xr.load_dataarray(f'../data/syn_piv_{_level}.nc')


if __name__ == '__main__':
    _create_synthetic_piv()
