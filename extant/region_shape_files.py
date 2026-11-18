# region number per grid cell: IMBIE basins on the continent, sea-ice sectors elsewhere
import numpy as np, xarray as xr, geopandas as gpd

REGION_CODES = {"East": 1, "West": 2, "Peninsula": 3, "Weddell": 4, "Indian": 5,
                "W Pacific": 6, "Ross": 7, "Amundsen-Bell.": 8}
IMBIE_SHP = "/work/scratch-pw4/aborow/shapefiles/Basins_IMBIE_Antarctica_v02.shp"


def open_imbie(path=IMBIE_SHP):
    """IMBIE 2016 basins (Rignot et al. 2016, NSIDC-0709)."""
    return gpd.read_file(path)


def imbie_regions(gdf, da):
    """East/West/Peninsula name per grid cell, NaN outside the polygons."""
    lon2, lat2 = np.meshgrid((da.lon + 180) % 360 - 180, da.lat)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon2.ravel(), lat2.ravel()), crs=4326).to_crs(gdf.crs)
    lab = gpd.sjoin(pts, gdf[["Regions", "geometry"]], how="left").groupby(level=0).Regions.first()
    return xr.DataArray(lab.values.reshape(lon2.shape), coords={"lat": da.lat, "lon": da.lon})


def ocean_sectors(da):
    """Sea-ice sector per longitude (Parkinson & Cavalieri 2012)."""
    l = (da.lon + 180) % 360 - 180
    return xr.where((l >= -60) & (l < 20), "Weddell",
           xr.where((l >= 20) & (l < 90), "Indian",
           xr.where((l >= 90) & (l < 160), "W Pacific",
           xr.where((l >= -130) & (l < -60), "Amundsen-Bell.", "Ross"))))


def region_ids(da, path=IMBIE_SHP):
    """Region number per grid cell; see REGION_CODES."""
    gdf = open_imbie(path)
    
    basin = imbie_regions(gdf, da)
    ocean_regions =  ocean_sectors(da)
    
    region = xr.where(basin.isin(["East", "West", "Peninsula"]), basin, ocean_regions)
    region_id = sum(i * (region == name) for name, i in REGION_CODES.items()).rename("region_id")

    #(t): One mask per region, named: True inside the region
    region_codes_da = xr.DataArray(list(REGION_CODES.values()), dims='region', coords={'region': list(REGION_CODES)})
    region_masks_da = region_id == region_codes_da
    
    return region_masks_da
