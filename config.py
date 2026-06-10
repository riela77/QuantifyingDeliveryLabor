"""
config.py - 라스트마일 배송 시뮬레이션 전역 설정
"""

# ── 터미널 (출발/복귀 거점) ─────────────────────────────────────
# 서울특별시 강서구 양천로 537
TERMINAL_LON  = 126.8635
TERMINAL_LAT  = 37.5396
TERMINAL_POS  = (TERMINAL_LON, TERMINAL_LAT)
TERMINAL_NAME = "강서 터미널 (양천로537)"

# ── 지역 설정 ──────────────────────────────────────────────────
REGIONS = {
    "sinlim": {
        "name":        "신림동",
        "label":       "평지형",
        "osm_query":   "Sillim-dong, Gwanak-gu, Seoul, South Korea",
        "center_lat":  37.4809,
        "center_lon":  126.9291,
        "graph_cache": "data/graph_sinlim.graphml",
        "color":       "blue",
        "route_color": "#378ADD",
    },
    "nangok": {
        "name":        "난곡동",
        "label":       "경사지형",
        "osm_query":   "Nangok-dong, Gwanak-gu, Seoul, South Korea",
        "center_lat":  37.4703,
        "center_lon":  126.9197,
        "graph_cache": "data/graph_nangok.graphml",
        "color":       "red",
        "route_color": "#D85A30",
    },
}

# ── GeoJSON 경로 ───────────────────────────────────────────────
ROAD_GEOJSON = "data2/관악구_smoothDEM.geojson"
CRS          = "EPSG:4326"

# ── 배송 설정 ─────────────────────────────────────────────────
# ★ 핵심: A* 실제 계산 건수. 300건은 초기화가 수십 분 걸림.
#   시뮬레이션은 NUM_SIM_HOUSES 건만 경로 계산,
#   나머지 (DAILY_DELIVERIES - NUM_SIM_HOUSES)건은 평균값으로 통계 추정.
DAILY_DELIVERIES = 300          # 하루 실제 배송량 (통계 표시용)
NUM_SIM_HOUSES   = 20           # 실제 A* 경로 계산 건수 (웹 시각화 대상)
NUM_HOUSES       = NUM_SIM_HOUSES

TRUCK_MIN_WIDTH  = 4.0          # 트럭 진입 최소 도로폭 (m)
PARK_RADIUS_M    = 300          # 주차지점 탐색 반경 (m)

# ── 속도/비용 상수 ────────────────────────────────────────────
TRUCK_SPEED_KMH   = 25.0
MOTO_SPEED_KMH    = 30.0
WALK_SPEED_KMH    = 4.0
TRUCK_COST_PER_KM = 800
MOTO_COST_PER_KM  = 400
FEE_PER_DELIVERY  = 2_500       # 건당 수수료 (원)

# ── 시뮬레이션 시간 단위 ──────────────────────────────────────
STEP_MIN = 1.0                  # 1 step = 1분

# ── 피로도 보정 ───────────────────────────────────────────────
FATIGUE_THRESHOLDS = [
    (0,       3 * 60, 3.0),
    (3 * 60,  6 * 60, 5.0),
    (6 * 60, 999_999, 7.0),
]
SLOPE_PENALTY = [
    (15, 2.0),
    (25, 3.0),
]

# ── GeoJSON slope 레이어 렌더링 옵션 ─────────────────────────
# True = 경사도 색상 도로 레이어 표시 (느림), False = 숨김 (빠름)
SHOW_SLOPE_LAYER = False

# ── 동 경계 필터 반경 (도 단위, ≈1.4km) ──────────────────────
REGION_RADIUS_DEG = 0.013