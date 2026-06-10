"""
model.py - 라스트마일 배송 시뮬레이션 모델

환경:
  G          : OSMnx 도로 그래프 (A* 경로탐색)
  road_gdf   : GeoJSON 도로 (min_width, slope, 4326 변환)

에이전트:
  HouseAgent     : 배송지 5곳
  TruckWalkAgent : 트럭+도보 시나리오
  MotoAgent      : 이륜차 시나리오
"""

import os
import random
import math
import numpy as np
import pandas as pd
import mesa
import osmnx as ox
import networkx as nx
import geopandas as gpd
from datetime import datetime
from mesa_geo import GeoSpace
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

from agents import HouseAgent, TruckWalkAgent, MotoAgent

# ── 설정 ──────────────────────────────────────────────────────────
START_LON       = 126.920401
START_LAT       = 37.470237
START_POS       = (START_LON, START_LAT)

GRAPH_CACHE     = "data/graph.graphml"
ROAD_GEOJSON    = "data2/관악구_smoothDEM.geojson"
NUM_HOUSES      = 10
CRS             = "EPSG:4326"
TRUCK_MIN_WIDTH = 4.0   # 트럭 진입 최소 도로폭 (m)
PARK_RADIUS_M   = 300   # 주차지점 탐색 반경 (m)

LOG_DIR         = "logs"  # CSV 로그 저장 디렉터리


# ── GeoJSON 도로 로드 및 전처리 ───────────────────────────────────
def load_road_gdf(path=ROAD_GEOJSON):
    """
    QGIS에서 5m 보간(Resampling) 완료된 최상급 GeoJSON 로드
    """
    print(f"  GeoJSON 도로 데이터 로드 중: {path}")
    gdf = gpd.read_file(path)

    if "slope_deg" in gdf.columns:
        gdf = gdf.rename(columns={"slope_deg": "slope"})
    else:
        print("  ⚠️ 주의: 데이터에 경사도(slope_deg) 컬럼이 없어 0.0으로 둡니다.")
        gdf["slope"] = 0.0

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    if gdf.crs.to_epsg() == 4326:
        gdf_4326 = gdf
        gdf_5179 = gdf.to_crs(epsg=5179)
    else:
        gdf_5179 = gdf
        gdf_4326 = gdf.to_crs(epsg=4326)

    gdf_4326["slope"] = gdf_4326["slope"].abs()
    gdf_5179["slope"] = gdf_5179["slope"].abs()

    print(f"  도로 구간: {len(gdf_4326)}개 | 리샘플링된 경사도: {gdf_4326['slope'].min():.2f}° ~ {gdf_4326['slope'].max():.2f}°")

    return gdf_4326, gdf_5179


# ── OSM 엣지에 min_width 조인 ─────────────────────────────────────
def enrich_graph_with_width(G, road_gdf_5179):
    print("  OSMnx 엣지에 도로폭 데이터 조인 중...")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False).to_crs(epsg=5179).reset_index()

    road_sub = road_gdf_5179[["geometry", "min_width"]].copy() if "min_width" in road_gdf_5179.columns else road_gdf_5179[["geometry"]].copy()
    if "min_width" not in road_sub.columns:
        road_sub["min_width"] = 999.0

    joined = gpd.sjoin_nearest(
        edges_gdf[["u", "v", "key", "geometry"]],
        road_sub,
        how="left",
        distance_col="_dist",
    )
    joined = joined.drop_duplicates(subset=["u", "v", "key"])

    for _, row in joined.iterrows():
        u, v, k = int(row["u"]), int(row["v"]), int(row["key"])
        mw = float(row["min_width"]) if not math.isnan(float(row["min_width"])) else 999.0
        if G.has_edge(u, v, k):
            G[u][v][k]["min_width"] = mw

    return G


# ── OSMnx 그래프 로드 ─────────────────────────────────────────────
def load_graph():
    if os.path.exists(GRAPH_CACHE):
        print("  캐시된 OSMnx 그래프 사용")
        return ox.load_graphml(GRAPH_CACHE)
    print("  OSMnx 그래프 다운로드 중 (난곡동)...")
    os.makedirs("data", exist_ok=True)
    G = ox.graph_from_place(
        "Nangok-dong, Gwanak-gu, Seoul, South Korea",
        network_type="all"
    )
    ox.save_graphml(G, GRAPH_CACHE)
    return G


# ── A* 경로탐색 ───────────────────────────────────────────────────
def astar_path(G, orig_lonlat, dest_lonlat):
    try:
        on = ox.distance.nearest_nodes(G, orig_lonlat[0], orig_lonlat[1])
        dn = ox.distance.nearest_nodes(G, dest_lonlat[0], dest_lonlat[1])
        route = nx.astar_path(
            G, on, dn,
            heuristic=lambda u, v: ox.distance.great_circle(
                G.nodes[u]["y"], G.nodes[u]["x"],
                G.nodes[v]["y"], G.nodes[v]["x"],
            ),
            weight="length",
        )
        return [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in route]
    except Exception as e:
        print(f"  A* 실패: {e} → 직선 경로")
        return [orig_lonlat, dest_lonlat]


# ── 주차지점 탐색 (min_width 필터) ───────────────────────────────
def find_park_pos(G, house_lonlat, road_gdf_4326):
    h_lon, h_lat = house_lonlat

    road_5179 = road_gdf_4326.to_crs(epsg=5179)
    h_pt_5179 = gpd.GeoSeries([Point(h_lon, h_lat)], crs="EPSG:4326").to_crs(epsg=5179).iloc[0]

    candidates = road_5179[road_5179["min_width"] >= TRUCK_MIN_WIDTH].copy() \
        if "min_width" in road_5179.columns else road_5179.copy()
    if candidates.empty:
        candidates = road_5179.copy()

    candidates["dist"] = candidates.geometry.distance(h_pt_5179)
    candidates = candidates[candidates["dist"] <= PARK_RADIUS_M]

    if candidates.empty:
        node = ox.distance.nearest_nodes(G, h_lon, h_lat)
        return (G.nodes[node]["x"], G.nodes[node]["y"])

    nearest_seg = candidates.nsmallest(1, "dist").iloc[0]
    near_pt_5179, _ = nearest_points(nearest_seg.geometry, h_pt_5179)

    near_pt_4326 = gpd.GeoSeries([near_pt_5179], crs="EPSG:5179").to_crs(epsg=4326).iloc[0]
    node = ox.distance.nearest_nodes(G, near_pt_4326.x, near_pt_4326.y)
    return (G.nodes[node]["x"], G.nodes[node]["y"])


# ── 경로 slope 추출 ───────────────────────────────────────────────
def get_slope_along_route(road_gdf_4326, route_lonlat):
    if road_gdf_4326 is None or "slope" not in road_gdf_4326.columns or not route_lonlat:
        return [0.0] * len(route_lonlat)

    pts_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in route_lonlat],
        crs="EPSG:4326",
    ).to_crs(epsg=5179)

    road_5179 = road_gdf_4326[["geometry", "slope"]].to_crs(epsg=5179)

    joined = gpd.sjoin_nearest(pts_gdf, road_5179, how="left")
    slopes = joined["slope"].fillna(0.0).tolist()
    return slopes


# ── 모델 ──────────────────────────────────────────────────────────
class DeliveryModel(mesa.Model):

    def __init__(self, scenario="truck_walk", seed=None):
        super().__init__(seed=seed)
        self.scenario_name  = scenario
        self.space          = GeoSpace(crs=CRS)
        self._step_count    = 0   # model 내부 step 카운터

        # ── 로그 저장소 ──────────────────────────────────────────
        self.sim_logs = []    # 1 step = 1분 단위 스냅샷

        print(f"[1/4] OSMnx 그래프 로드 중...")
        self.G = load_graph()

        print(f"[2/4] GeoJSON 도로 데이터 로드 및 변환 중...")
        self.road_gdf_4326, road_gdf_5179 = load_road_gdf()

        self.G = enrich_graph_with_width(self.G, road_gdf_5179)

        nodes_gdf   = ox.graph_to_gdfs(self.G, edges=False)
        bounds      = nodes_gdf.total_bounds
        self.center = (
            (bounds[1] + bounds[3]) / 2,
            (bounds[0] + bounds[2]) / 2,
        )

        print(f"[3/4] 배송지 {NUM_HOUSES}곳 생성 중...")
        self.houses = []
        all_nodes   = list(self.G.nodes)
        random.seed(seed)
        used = set()
        while len(self.houses) < NUM_HOUSES:
            node = random.choice(all_nodes)
            if node in used:
                continue
            used.add(node)
            lon   = self.G.nodes[node]["x"]
            lat   = self.G.nodes[node]["y"]
            house = HouseAgent(self, Point(lon, lat), CRS)
            self.space.add_agents(house)
            self.houses.append(house)

        print(f"[4/4] 배달원 에이전트 생성 및 경로 계산 중...")
        start_geom = Point(START_LON, START_LAT)

        if scenario == "truck_walk":
            agent = TruckWalkAgent(self, start_geom, CRS)
            segs  = self._build_truck_segments(self.houses)
            agent.set_segments(segs)
        else:
            agent = MotoAgent(self, start_geom, CRS)
            segs  = self._build_moto_segments(self.houses)
            agent.set_segments(segs)

        self.space.add_agents(agent)
        self.delivery_agent = agent

        print(f"✅ 초기화 완료 [{scenario}]")
        print(f"   배송지: {NUM_HOUSES}곳 | 총 {agent.delivered_kg:.1f}kg")
        print(f"   예상 시간: {agent.total_time_min:.1f}분 | 비용: {agent.total_cost_won:,.0f}원")

    # ── 세그먼트 빌더 ──────────────────────────────────────────────
    def _build_truck_segments(self, houses):
        segments = []
        cur_pos  = START_POS
        for house in houses:
            h_pos      = (house.geometry.x, house.geometry.y)
            park_pos   = find_park_pos(self.G, h_pos, self.road_gdf_4326)
            rt_truck   = astar_path(self.G, cur_pos,  park_pos)
            rt_walk    = astar_path(self.G, park_pos, h_pos)
            slope_walk = get_slope_along_route(self.road_gdf_4326, rt_walk)
            segments.append({
                "truck":      rt_truck,
                "walk":       rt_walk,
                "slope_walk": slope_walk,
                "park_pos":   park_pos,
                "house":      house,
            })
            cur_pos = h_pos
        return segments

    def _build_moto_segments(self, houses):
        segments = []
        cur_pos  = START_POS
        for house in houses:
            h_pos      = (house.geometry.x, house.geometry.y)
            route      = astar_path(self.G, cur_pos, h_pos)
            slope_list = get_slope_along_route(self.road_gdf_4326, route)
            segments.append({
                "route":      route,
                "slope_list": slope_list,
                "house":      house,
            })
            cur_pos = h_pos
        return segments

    # ── step() ── 매 1분마다 호출 ─────────────────────────────────
    def step(self):
        self.delivery_agent.step()
        for house in self.houses:
            house.step()

        self._step_count += 1
        ag = self.delivery_agent

        # ── 1분 단위 스냅샷 로그 기록 ────────────────────────────
        is_truck_agent = isinstance(ag, TruckWalkAgent)
        log_entry = {
            "step_min":        self._step_count,          # 경과 시간(분)
            "scenario":        self.scenario_name,         # 시나리오 이름
            "phase":           ag.phase,                   # truck / walk / riding / done
            "current_slope":   round(ag.current_slope, 3), # 현재 경사도(도)
            "carried_kg":      round(ag.carried_kg, 2),    # 현재 들고 있는 화물 무게(kg)
            "delivered_count": ag.delivered_count,          # 누적 배송 완료 건수
            "steep_crossings": ag.steep_crossings,          # 험지 돌파 횟수
            "lon":             round(ag.geometry.x, 6),     # 현재 경도
            "lat":             round(ag.geometry.y, 6),     # 현재 위도
        }

        if is_truck_agent:
            log_entry["cumul_walk_km"]  = round(ag.cumul_walk_m  / 1000, 4)
            log_entry["cumul_truck_km"] = round(ag.cumul_truck_m / 1000, 4)
        else:
            log_entry["cumul_dist_km"]  = round(ag.cumul_dist_m  / 1000, 4)

        self.sim_logs.append(log_entry)

    # ── 시뮬레이션 종료 시 CSV 저장 ──────────────────────────────
    def save_logs(self):
        """
        호출 시점에 sim_logs를 CSV로 저장합니다.
        app.py의 run_simulation() 루프에서 phase == 'done' 감지 후 호출.

        저장 경로: logs/delivery_log_{scenario}_{timestamp}.csv
        인코딩: utf-8-sig (Excel 한글 깨짐 방지)
        """
        if not self.sim_logs:
            print("  ⚠️ 로그 데이터가 없습니다.")
            return None

        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = os.path.join(LOG_DIR, f"delivery_log_{self.scenario_name}_{timestamp}.csv")

        df = pd.DataFrame(self.sim_logs)

        # ── 파생 컬럼 추가 (분석 편의) ──────────────────────────
        # 도보 중인 step만 골라 slope * carried_kg → '경사 부하 지수'
        df["slope_load_idx"] = (
            df["current_slope"].abs() * df["carried_kg"]
        ).round(3)

        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"💾 로그 저장 완료: {filename} (총 {len(df)} steps / {self._step_count}분)")
        print(f"   도보 phase 비율: {(df['phase']=='walk').mean()*100:.1f}%")
        print(f"   평균 경사도(도보): {df[df['phase']=='walk']['current_slope'].mean():.2f}°")
        print(f"   최대 경사 부하: {df['slope_load_idx'].max():.1f} (°×kg)")
        return filename