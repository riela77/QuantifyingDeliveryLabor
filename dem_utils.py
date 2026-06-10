"""
DEM 유틸리티
- 90m DEM → 5m bilinear 보간 업샘플링
- 도로 GeoDataFrame에 elevation + slope_segments 계산
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def upsample_dem(src, target_res=5.0):
    """90m DEM → target_res(m) bilinear 업샘플링"""
    scale      = src.res[0] / target_res
    new_height = int(src.height * scale)
    new_width  = int(src.width  * scale)
    new_transform = src.transform * src.transform.scale(
        src.width / new_width, src.height / new_height
    )
    data = np.empty((new_height, new_width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=data,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=new_transform,
        dst_crs=src.crs,
        resampling=Resampling.bilinear,
    )
    return data, new_transform, src.crs, src.nodata


def sample_elev(data, transform, nodata, x, y):
    try:
        col, row = ~transform * (x, y)
        r, c = int(row), int(col)
        if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
            v = float(data[r, c])
            return 0.0 if (nodata is not None and v == nodata) else v
        return 0.0
    except Exception:
        return 0.0


def densify_line(line, step):
    pts = [line.interpolate(d) for d in np.arange(0, line.length, step)]
    pts.append(line.interpolate(line.length))
    return pts


def build_road_layer(road_gdf, dem_path, densify_step=5.0):
    """
    도로 GeoDataFrame에 배경 속성 추가
    - elevation     : 평균 고도
    - slope         : 평균 경사도
    - slope_segments: 5m 구간별 경사도 리스트
    도로는 에이전트가 아닌 순수 배경 데이터로 사용
    """
    print(f"  DEM 로드: {dem_path}")
    with rasterio.open(dem_path) as src:
        print(f"  원본 해상도: {src.res[0]}m → {densify_step}m bilinear 업샘플링")
        data, transform, crs, nodata = upsample_dem(src, target_res=densify_step)
        print(f"  업샘플링 완료: {data.shape}")

    road_proj = road_gdf.to_crs(crs)
    elevations, slopes, seg_list = [], [], []

    for geom in road_proj.geometry:
        try:
            length = geom.length
            if length < densify_step:
                x0, y0 = geom.coords[0]
                e = sample_elev(data, transform, nodata, x0, y0)
                elevations.append(e); slopes.append(0.0); seg_list.append([0.0])
                continue

            pts   = densify_line(geom, densify_step)
            elevs = [sample_elev(data, transform, nodata, p.x, p.y) for p in pts]
            elevations.append(float(np.mean(elevs)))

            segs = [
                round(float(np.degrees(np.arctan2(
                    elevs[i+1] - elevs[i], densify_step
                ))), 3)
                for i in range(len(elevs) - 1)
            ]
            seg_list.append(segs if segs else [0.0])
            slopes.append(round(float(np.mean(segs)), 3) if segs else 0.0)

        except Exception:
            elevations.append(0.0); slopes.append(0.0); seg_list.append([0.0])

    road_gdf = road_gdf.copy()
    road_gdf["elevation"]      = elevations
    road_gdf["slope"]          = slopes
    road_gdf["slope_segments"] = seg_list

    elev_min, elev_max = float(min(elevations)), float(max(elevations))
    print(f"  고도: {elev_min:.1f}m ~ {elev_max:.1f}m")
    print(f"  경사도: {min(slopes):.2f}° ~ {max(slopes):.2f}°")

    result = road_gdf.to_crs(epsg=4326)
    result.attrs["elev_min"] = elev_min
    result.attrs["elev_max"] = elev_max
    return result


def get_road_slope_at(road_gdf_proj, pos_proj, densify_step=5.0):
    """트럭 위치의 가장 가까운 도로 5m 구간 slope 반환"""
    try:
        dists       = road_gdf_proj.geometry.distance(pos_proj)
        nearest_idx = dists.idxmin()
        geom        = road_gdf_proj.geometry[nearest_idx]
        segs        = road_gdf_proj["slope_segments"][nearest_idx]
        if not segs or len(segs) == 1:
            return segs[0] if segs else 0.0
        dist_along = geom.project(pos_proj)
        idx = max(0, min(int(dist_along // densify_step), len(segs) - 1))
        return segs[idx]
    except Exception:
        return 0.0