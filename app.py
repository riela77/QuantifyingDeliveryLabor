"""
app.py - 라스트마일 배송 시뮬레이션 (Solara + ipyleaflet)

실행: solara run app.py
"""

import time
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import solara
import ipyleaflet as ipl
import ipywidgets as widgets
from pathlib import Path
from model import DeliveryModel, NUM_HOUSES, NUM_AGENTS, START_LAT, START_LON
from agents import TruckWalkAgent, MotoAgent, STEEP_THRESHOLD

# 터미널 출발지
TERMINAL_NAME    = "구로 터미널"
TERMINAL_ADDRESS = "서울특별시 구로구 경인로 110"

# 에이전트별 마커 색상
AGENT_COLORS = ["cadetblue", "purple", "darkred", "darkgreen", "orange", "darkblue", "pink"]


# ── 범례 HTML (WidgetControl용) ──────────────────────────────────
_LEGEND_HTML = """
<div class="lm-wc-legend">
  <div class="lm-leg"><span class="lm-ll" style="background:#238636"></span>평지 &lt;15°</div>
  <div class="lm-leg"><span class="lm-ll" style="background:#d29922"></span>경사 15~25°</div>
  <div class="lm-leg"><span class="lm-ll" style="background:#f85149"></span>급경사 25°+</div>
  <div style="border-top:0.5px solid #21262d;margin:4px 0"></div>
  <div class="lm-leg"><span class="lm-ldot" style="background:#388bfd"></span>차량 이동</div>
  <div class="lm-leg"><span class="lm-ldot" style="background:#f0883e"></span>도보 배송</div>
  <div class="lm-leg"><span class="lm-ldot" style="background:#2ea043"></span>배송 완료</div>
</div>
"""


# ── 우측 패널 HTML 생성 함수 ────────────────────────────────────
def _panel_html(ag, step, houses, walk_km, truck_km,
                carried_kg, steep, max_hours, log_path, scenario,
                total_orders, num_workers, steep_km, alley_pct,
                agents):
    is_truck  = isinstance(ag, TruckWalkAgent)
    phase     = ag.phase
    delivered = ag.delivered_count
    slope     = ag.current_slope
    over_time = step > max_hours * 60
    warn_time = step > max_hours * 60 * 0.8
    pct       = int(delivered / max(len(houses) // NUM_AGENTS, 1) * 100)
    hh, mm    = step // 60, step % 60
    sl_pct    = min(abs(slope) / 30 * 100, 100)
    sl_color  = "#f85149" if abs(slope) >= 25 else "#d29922" if abs(slope) >= 15 else "#238636"

    drv_cls   = "lm-drv over" if over_time else "lm-drv warn" if warn_time else "lm-drv"
    badge_cls = ("lm-badge over"  if over_time  else
                 "lm-badge done"  if phase=="done" else
                 "lm-badge walk"  if phase=="walk" else "lm-badge truck")
    badge_txt = "초과" if over_time else \
                "DONE" if phase=="done" else \
                "WALK" if phase=="walk" else "TRUCK"
    bar_cls   = "lm-dpfg over" if over_time else "lm-dpfg warn" if warn_time else "lm-dpfg"

    time_cls   = "lm-overtime-blink" if over_time else ""
    time_color = "#f85149" if over_time else "#e6edf3"

    # 전체 기사 요약 카드
    active_count = sum(1 for a in agents if a.phase not in ("done", "idle"))
    done_count   = sum(1 for a in agents if a.phase == "done")
    walk_count   = sum(1 for a in agents if a.phase == "walk")
    over_count   = sum(1 for a in agents if step > max_hours * 60)

    # 기사 목록
    driver_list = ""
    for i, a in enumerate(agents):
        a_over  = step > max_hours * 60
        a_warn  = step > max_hours * 60 * 0.8
        a_cls   = "lm-drv over" if a_over else "lm-drv warn" if a_warn else "lm-drv"
        a_phase = a.phase.upper()
        a_color = "#f85149" if a_over else "#d29922" if a_warn else "#8b949e"
        a_pct   = int(a.delivered_count / max(len(houses) // NUM_AGENTS, 1) * 100)
        a_bar   = "lm-dpfg over" if a_over else "lm-dpfg warn" if a_warn else "lm-dpfg"
        wkm     = round(a.cumul_walk_m / 1000, 2) if hasattr(a, "cumul_walk_m") else round(a.cumul_dist_m / 1000, 2)
        driver_list += f"""
<div class="{a_cls}" style="margin-bottom:5px">
  <div class="lm-drv-head">
    <span class="lm-drv-name">#{i+1:02d} 배송기사</span>
    <span class="lm-badge" style="color:{a_color};border-color:{a_color}">{a_phase}</span>
  </div>
  <div class="lm-drow"><span>도보</span><span>{wkm:.2f} km</span></div>
  <div class="lm-drow"><span>완료</span><span>{a.delivered_count}건</span></div>
  <div class="lm-dpbg"><div class="{a_bar}" style="width:{a_pct}%"></div></div>
</div>"""

    truck_card = (
        f'<div class="lm-lc"><div class="lm-lc-lbl">차량 이동</div>'
        f'<div class="lm-lc-val">{truck_km:.2f} km</div></div>'
        if is_truck else ""
    )
    log_line = (
        f'<div style="font-size:9px;color:var(--grn-hi);font-family:var(--mono);margin-top:4px">💾 {log_path}</div>'
        if log_path else ""
    )
    all_done  = all(a.phase == "done" for a in agents)
    done_block = (
        f'<div class="lm-done">🎉 모든 배송 완료<br>'
        f'<span style="font-size:9px;color:var(--t3)">총 {step}분 · 도보 {walk_km:.2f}km · 험지 {steep}회</span>'
        f'{log_line}</div>'
    ) if all_done else ""

    ot_block = '<div class="lm-ot">⛔ 근무시간 초과</div>' if over_time else ""

    walk_cls  = "lm-kpi-val danger" if walk_km > 5  else "lm-kpi-val warn" if walk_km > 2  else "lm-kpi-val"
    steep_cls = "lm-kpi-val danger" if steep_km > 1 else "lm-kpi-val warn" if steep_km > 0.3 else "lm-kpi-val"
    alley_cls = "lm-kpi-val danger" if alley_pct > 50 else "lm-kpi-val warn" if alley_pct > 25 else "lm-kpi-val"

    return f"""
<div class="lm-panel">
  <input type="radio" name="lm-tab" id="tab-status" checked>
  <input type="radio" name="lm-tab" id="tab-params">
  <input type="radio" name="lm-tab" id="tab-env">

  <div class="lm-tabs">
    <label class="lm-tab" for="tab-status">📊 현황</label>
    <label class="lm-tab" for="tab-params">⚙️ 물동량</label>
    <label class="lm-tab" for="tab-env">🗺️ 환경</label>
  </div>

  <!-- ── 현황 탭 ── -->
  <div class="lm-pane" id="lm-pane-status">

    <div class="lm-mgrid">
      <div class="lm-metric">
        <div class="lm-mv" style="color:#2ea043">{active_count}</div>
        <div class="lm-ml">🚛 가동 중</div>
      </div>
      <div class="lm-metric">
        <div class="lm-mv" style="color:#f85149">{over_count}</div>
        <div class="lm-ml">⛔ 시간초과</div>
      </div>
      <div class="lm-metric">
        <div class="lm-mv" style="color:#388bfd">{done_count}</div>
        <div class="lm-ml">✅ 완료 기사</div>
      </div>
      <div class="lm-metric">
        <div class="lm-mv" style="color:#f0883e">{walk_count}</div>
        <div class="lm-ml">🚶 도보 中</div>
      </div>
    </div>

    {ot_block}

    <div style="background:#0d1117;border:0.5px solid #30363d;border-radius:8px;
                padding:9px 12px;display:flex;justify-content:space-between;align-items:center">
      <span style="font-family:var(--mono);font-size:10px;color:#8b949e">⏱ 경과 시간</span>
      <span class="{time_cls}" style="font-family:var(--mono);font-size:18px;font-weight:600;color:{time_color}">{hh:02d}:{mm:02d}</span>
    </div>

    <div class="lm-sec">노동 강도 핵심 지표</div>
    <div class="lm-kpi-grid">
      <div class="lm-kpi">
        <span class="lm-kpi-lbl">🚶 총 도보 거리</span>
        <span class="{walk_cls}">{walk_km:.2f} km</span>
      </div>
      <div class="lm-kpi">
        <span class="lm-kpi-lbl">⛰ 급경사 구간</span>
        <span class="{steep_cls}">{steep_km:.2f} km</span>
      </div>
      <div class="lm-kpi">
        <span class="lm-kpi-lbl">🏘 골목 비율</span>
        <span class="{alley_cls}">{alley_pct:.1f} %</span>
      </div>
    </div>

    <div class="lm-slopebox">
      <div class="lm-sloperow">
        <span style="font-size:10px;color:#8b949e;font-family:var(--mono)">현재 경사도 (#01)</span>
        <span style="font-family:var(--mono);font-weight:600;font-size:13px;color:#e6edf3">{abs(slope):.1f}°</span>
      </div>
      <div class="lm-pwrap">
        <div class="lm-pfill" style="width:{sl_pct:.1f}%;background:{sl_color};transition:width .5s,background .5s"></div>
      </div>
    </div>

    <div class="lm-sec">기사별 현황</div>
    {driver_list}
    {done_block}
  </div>

  <!-- ── 물동량 탭 ── -->
  <div class="lm-pane" id="lm-pane-params">
    <div class="lm-sec">물동량 설정</div>
    <div class="lm-input-box">
      <div class="lm-input-row">
        <span>기본 물동량</span>
        <span class="lm-input-val">{total_orders}건</span>
      </div>
      <div class="lm-input-row">
        <span>할당 기사 수</span>
        <span class="lm-input-val">{num_workers}명</span>
      </div>
      <div class="lm-input-row">
        <span>기사당 물량</span>
        <span class="lm-input-val" style="color:{'#f85149' if total_orders//max(num_workers,1) > 150 else '#d29922' if total_orders//max(num_workers,1) > 100 else '#e6edf3'}">{total_orders // max(num_workers, 1)}건</span>
      </div>
    </div>
    <div style="font-size:9px;color:var(--t3);font-family:var(--mono);padding:0 2px">
      ※ 물량 추가는 컨트롤 바의 [+10건] [+50건] 버튼을 사용하세요
    </div>

    <div class="lm-sec" style="margin-top:4px">피로도 모델</div>
    <div class="lm-info">
      경사 &lt;15° → 기본 속도<br>
      경사 15~25° → <span class="y">×0.5 감속</span><br>
      경사 25°+ &nbsp;→ <span class="r">×0.3 감속</span><br>
      협로 (폭≤4m) → <span class="r">도보 강제전환</span>
    </div>

    <div class="lm-sec" style="margin-top:4px">예상 결과 (#01 기준)</div>
    <div class="lm-info">
      총 화물: <span style="color:#e6edf3">{ag.delivered_kg:.1f} kg</span><br>
      예상 시간: <span style="color:#e6edf3">{ag.total_time_min:.0f}분</span><br>
      예상 비용: <span style="color:#e6edf3">{ag.total_cost_won:,.0f}원</span>
    </div>
  </div>

  <!-- ── 환경 탭 ── -->
  <div class="lm-pane" id="lm-pane-env">
    <div class="lm-sec">출발 터미널</div>
    <div class="lm-info">
      터미널명: <span style="color:#e6edf3">{TERMINAL_NAME}</span><br>
      주소: <span style="color:#e6edf3">{TERMINAL_ADDRESS}</span><br>
      좌표계: <span style="color:#e6edf3">EPSG 4326</span><br>
      DEM: <span style="color:#e6edf3">90m → 5m 보간</span>
    </div>
    <div class="lm-sec" style="margin-top:4px">도로 범례</div>
    <div style="font-size:10px;font-family:var(--mono);color:#8b949e;line-height:2.2">
      <div class="lm-leg"><span class="lm-ll" style="background:#238636"></span>평지 &lt;15°</div>
      <div class="lm-leg"><span class="lm-ll" style="background:#d29922"></span>경사 15~25°</div>
      <div class="lm-leg"><span class="lm-ll" style="background:#f85149"></span>급경사 25°+</div>
      <div style="border-top:0.5px solid #21262d;margin:5px 0"></div>
      <div class="lm-leg"><span class="lm-ldot" style="background:#388bfd"></span>차량 이동</div>
      <div class="lm-leg"><span class="lm-ldot" style="background:#f0883e"></span>도보 배송</div>
      <div class="lm-leg"><span class="lm-ldot" style="background:#2ea043"></span>배송 완료</div>
    </div>
  </div>
</div>
"""


# ── Page 컴포넌트 ────────────────────────────────────────────────
@solara.component
def Page():
    css_path = Path("public/style.css")
    if css_path.exists():
        solara.Style(css_path.read_text(encoding="utf-8"))

    scenario, set_scenario = solara.use_state("truck_walk")
    max_hours = 8

    model = solara.use_memo(
        lambda: DeliveryModel(scenario=scenario),
        dependencies=[scenario],
    )
    map_obj = solara.use_memo(
        lambda: ipl.Map(
            center=model.center,
            zoom=15,
            scroll_wheel_zoom=True,
            prefer_canvas=True,
            layout={'height': '780px', 'width': '100%'}
        ),
        dependencies=[scenario],
    )
    marker_dict = solara.use_memo(lambda: {}, dependencies=[scenario])

    running         = solara.use_reactive(False)
    step_count      = solara.use_reactive(0)
    speed           = solara.use_reactive(10)
    log_saved       = solara.use_reactive(False)
    log_path        = solara.use_reactive("")
    alert_msg       = solara.use_reactive("")
    live_walk_km    = solara.use_reactive(0.0)
    live_truck_km   = solara.use_reactive(0.0)
    live_carried_kg = solara.use_reactive(0.0)
    live_steep      = solara.use_reactive(0)
    live_steep_km   = solara.use_reactive(0.0)
    live_alley_pct  = solara.use_reactive(0.0)
    total_orders    = solara.use_reactive(170)
    num_workers     = solara.use_reactive(NUM_AGENTS)

    status_widget = solara.use_memo(
        lambda: widgets.HTML(value='<div class="lm-wc-status">▶ 실행을 눌러 시작하세요</div>'),
        dependencies=[scenario],
    )
    alert_widget = solara.use_memo(
        lambda: widgets.HTML(value='<div class="lm-wc-alert" style="display:none"></div>'),
        dependencies=[scenario],
    )
    legend_widget = solara.use_memo(
        lambda: widgets.HTML(value=_LEGEND_HTML),
        dependencies=[scenario],
    )

    # ── 지도 초기화 ───────────────────────────────────────────────
    def init_map():
        for key in list(marker_dict.keys()):
            del marker_dict[key]
        for layer in [l for l in map_obj.layers if not isinstance(l, ipl.TileLayer)]:
            map_obj.remove(layer)
        for ctrl in list(map_obj.controls):
            if hasattr(ctrl, 'widget'):
                map_obj.remove(ctrl)

        # 경로 폴리라인 (모든 에이전트)
        for ag in model.delivery_agents:
            if isinstance(ag, TruckWalkAgent):
                for seg in ag.segments:
                    if seg["truck"]:
                        map_obj.add(ipl.Polyline(
                            locations=[[lat, lon] for lon, lat in seg["truck"]],
                            color="#388bfd", weight=2, opacity=0.45))
                    if seg["walk"]:
                        map_obj.add(ipl.Polyline(
                            locations=[[lat, lon] for lon, lat in seg["walk"]],
                            color="#f0883e", weight=2, opacity=0.55, dash_array="6 4"))
            else:
                for seg in ag.segments:
                    if seg["route"]:
                        map_obj.add(ipl.Polyline(
                            locations=[[lat, lon] for lon, lat in seg["route"]],
                            color="#f0883e", weight=2, opacity=0.5))

        # 도로 레이어
        road_gdf  = model.road_gdf_4326
        slope_p95 = float(np.percentile(road_gdf["slope"].dropna(), 95)) \
                    if "slope" in road_gdf.columns else 5.0
        slope_max = max(slope_p95, 1.0)
        cmap = matplotlib.colormaps["RdYlGn_r"]
        norm = mcolors.Normalize(vmin=0, vmax=slope_max)
        def style_cb(feature):
            s = min(feature["properties"].get("slope", 0), slope_max)
            c = mcolors.to_hex(cmap(norm(s)))
            w = float(feature["properties"].get("min_width", 2.0))
            return {"color": c, "weight": max(1.5, min(w * 0.5, 5)), "opacity": 0.75}
        map_obj.add(ipl.GeoJSON(data=road_gdf.__geo_interface__, style_callback=style_cb))

        # 터미널 마커
        map_obj.add(ipl.Marker(
            location=(START_LAT, START_LON),
            icon=ipl.AwesomeIcon(name="industry", marker_color="blue"),
            title=f"{TERMINAL_NAME} (출발지)", draggable=False))

        # 배송지 마커
        for i, house in enumerate(model.houses):
            m = ipl.CircleMarker(
                location=(house.geometry.y, house.geometry.x),
                radius=7, color="#484f58", fill_color="#484f58",
                fill_opacity=0.9, weight=2,
                title=f"배송지 {i+1} | {house.pkg_kg}kg")
            marker_dict[f"house_{house.unique_id}"] = m
            map_obj.add(m)

        # 에이전트 마커 (각기 다른 색상)
        for i, ag in enumerate(model.delivery_agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            name  = "truck" if isinstance(ag, TruckWalkAgent) else "motorcycle"
            icon  = ipl.AwesomeIcon(name=name, marker_color=color)
            m = ipl.Marker(
                location=(ag.geometry.y, ag.geometry.x),
                icon=icon, draggable=False, title=f"배송기사 #{i+1}")
            marker_dict[f"agent_{i}"] = m
            map_obj.add(m)

        # WidgetControl
        map_obj.add(ipl.WidgetControl(widget=alert_widget,  position="topleft"))
        map_obj.add(ipl.WidgetControl(widget=status_widget, position="bottomleft"))
        map_obj.add(ipl.WidgetControl(widget=legend_widget, position="bottomright"))

    solara.use_effect(init_map, [scenario])

    def _calc_alley_pct():
        try:
            road_gdf = model.road_gdf_4326
            if "min_width" not in road_gdf.columns:
                return 0.0
            total = len(road_gdf)
            alley = (road_gdf["min_width"] <= 4).sum()
            return round(alley / max(total, 1) * 100, 1)
        except Exception:
            return 0.0

    def _set_alert(msg: str):
        alert_msg.set(msg)
        alert_widget.value = f'<div class="lm-wc-alert">⚠ {msg}</div>'

    # ── 시뮬레이션 루프 ───────────────────────────────────────────
    def run_simulation():
        if not running.value:
            return

        live_alley_pct.set(_calc_alley_pct())
        ag0       = model.delivery_agent  # 대표 에이전트 (#01)
        prev_phase = ag0.phase

        while running.value:
            # 모든 에이전트 완료 체크
            all_done = all(a.phase == "done" for a in model.delivery_agents)
            if all_done:
                if not log_saved.value:
                    path = model.save_logs()
                    log_saved.set(True); log_path.set(path or "")
                    _set_alert("💾 로그 저장 완료")
                status_widget.value = '<div class="lm-wc-status">🎉 모든 배송 완료!</div>'
                running.set(False); break

            model.step()

            # 마커 위치 업데이트 (전체 에이전트)
            for i, ag in enumerate(model.delivery_agents):
                key = f"agent_{i}"
                if key in marker_dict:
                    marker_dict[key].location = (ag.geometry.y, ag.geometry.x)
                    if isinstance(ag, TruckWalkAgent):
                        color = AGENT_COLORS[i % len(AGENT_COLORS)] if ag.phase == "truck" else "orange"
                        name  = "truck" if ag.phase == "truck" else "male"
                        marker_dict[key].icon = ipl.AwesomeIcon(name=name, marker_color=color)

            # 배송지 완료 색상
            for house in model.houses:
                key = f"house_{house.unique_id}"
                if key in marker_dict and house.visited:
                    marker_dict[key].color = "#2ea043"
                    marker_dict[key].fill_color = "#2ea043"

            # 알림 (대표 에이전트 기준)
            cur = ag0.phase
            if cur != prev_phase:
                if cur == "walk":
                    seg = ag0.current_segment
                    kg  = seg["house"].pkg_kg if seg else 0
                    _set_alert(f"차량 진입 불가 — 도보 전환 ({kg}kg)")
                elif cur == "truck":
                    _set_alert("다음 배송지로 차량 이동 중")
                prev_phase = cur
            elif cur == "walk" and abs(ag0.current_slope) >= STEEP_THRESHOLD:
                _set_alert(f"급경사 구간 진입 ({ag0.current_slope:.1f}°)")

            # 지표 갱신 (대표 에이전트 기준)
            sc = step_count.value + 1
            if isinstance(ag0, TruckWalkAgent):
                live_walk_km.set(round(ag0.cumul_walk_m  / 1000, 3))
                live_truck_km.set(round(ag0.cumul_truck_m / 1000, 3))
            else:
                live_walk_km.set(round(ag0.cumul_dist_m / 1000, 3))
            live_carried_kg.set(ag0.carried_kg)
            live_steep.set(ag0.steep_crossings)
            live_steep_km.set(round(ag0.steep_crossings * 0.05, 2))
            step_count.set(sc)
            status_widget.value = f'<div class="lm-wc-status">▶ 실행 중 &nbsp;|&nbsp; Step {sc}</div>'

            time.sleep(1 / max(speed.value, 1))

    solara.use_thread(run_simulation, [running.value])

    def on_run():
        running.set(not running.value)
        if not running.value:
            status_widget.value = '<div class="lm-wc-status">⏸ 일시정지</div>'

    def on_step():
        ag0 = model.delivery_agent
        if not all(a.phase == "done" for a in model.delivery_agents):
            model.step()
            for i, ag in enumerate(model.delivery_agents):
                key = f"agent_{i}"
                if key in marker_dict:
                    marker_dict[key].location = (ag.geometry.y, ag.geometry.x)
            if isinstance(ag0, TruckWalkAgent):
                live_walk_km.set(round(ag0.cumul_walk_m  / 1000, 3))
                live_truck_km.set(round(ag0.cumul_truck_m / 1000, 3))
            else:
                live_walk_km.set(round(ag0.cumul_dist_m / 1000, 3))
            live_carried_kg.set(ag0.carried_kg)
            live_steep.set(ag0.steep_crossings)
            live_steep_km.set(round(ag0.steep_crossings * 0.05, 2))
            sc = step_count.value + 1
            step_count.set(sc)
            status_widget.value = f'<div class="lm-wc-status">▶ Step {sc}</div>'

    def on_reset():
        set_scenario(scenario)
        log_saved.set(False); log_path.set(""); step_count.set(0)
        live_walk_km.set(0.0); live_truck_km.set(0.0)
        live_carried_kg.set(0.0); live_steep.set(0)
        live_steep_km.set(0.0); live_alley_pct.set(0.0)
        alert_msg.set("")
        status_widget.value = '<div class="lm-wc-status">▶ 실행을 눌러 시작하세요</div>'
        alert_widget.value  = '<div class="lm-wc-alert" style="display:none"></div>'

    def on_log():
        path = model.save_logs(); log_path.set(path or "")
        _set_alert(f"💾 로그 저장: {path}")

    def on_add_10():
        total_orders.set(total_orders.value + 10)

    def on_add_50():
        total_orders.set(total_orders.value + 50)

    ag0 = model.delivery_agent
    solara.Title("LastMile Labor")

    # ── 헤더 ─────────────────────────────────────────────────────
    hh_e, mm_e = step_count.value // 60, step_count.value % 60
    active_n = sum(1 for a in model.delivery_agents if a.phase not in ("done", "idle"))
    solara.HTML("div", unsafe_innerHTML=f"""
    <div class="lm-header">
      <div class="lm-logo"><span class="br">[</span>LM<span class="br">]</span>&nbsp;LastMile Labor</div>
      <div class="lm-logo-sep"></div>
      <div class="lm-logo-sub">배송 노동 가시화 · ABM · 구로구 경인로</div>
      <div class="lm-ticker">
        <span class="sv">{step_count.value}</span>
        <span style="color:#484f58">step</span>
        <span class="ph">{active_n}/{NUM_AGENTS}</span>
        &nbsp;|&nbsp;
        <span class="sv">{hh_e:02d}:{mm_e:02d}</span>
      </div>
    </div>""")

    # ── 컨트롤 바 ─────────────────────────────────────────────────
    run_lbl = "⏸ 일시정지" if running.value else "▶ 자동재생"
    run_style = (
        "background:#da3633!important;color:#fff!important;"
        if running.value else
        "background:#1f6feb!important;color:#fff!important;"
    )
    btn_base = (
        "border-radius:7px!important;font-family:monospace!important;"
        "font-size:12px!important;text-transform:none!important;"
        "letter-spacing:0!important;box-shadow:none!important;"
        "height:32px!important;min-height:32px!important;"
        "border:0.5px solid #30363d!important;"
    )
    btn_sec = "background:#21262d!important;color:#c9d1d9!important;" + btn_base
    btn_add = "background:#21262d!important;color:#d29922!important;border-color:#d29922!important;" + btn_base

    with solara.Row(style=(
        "background:#161b22!important;border-bottom:1px solid #30363d;"
        "padding:7px 16px;gap:8px;flex-wrap:wrap;align-items:center;"
    )):
        solara.Button(run_lbl,         on_click=on_run,   style=run_style + btn_base)
        solara.Button("▷ 1 Step",     on_click=on_step,  style=btn_sec, disabled=running.value)
        solara.Button("↺ 초기화",     on_click=on_reset, style=btn_sec, disabled=running.value)
        solara.Button("💾 로그 저장",  on_click=on_log,   style=btn_sec)
        solara.HTML("span", unsafe_innerHTML='<span style="width:1px;height:20px;background:#30363d;display:inline-block;margin:0 4px"></span>')
        solara.Button(f"+10건 ({total_orders.value + 10})", on_click=on_add_10, style=btn_add)
        solara.Button(f"+50건 ({total_orders.value + 50})", on_click=on_add_50, style=btn_add)
        solara.HTML("span", unsafe_innerHTML=(
            f'<span style="font-family:monospace;font-size:11px;color:#8b949e;">'
            f'물동량&nbsp;<span style="color:#d29922;font-weight:600">{total_orders.value}</span>건'
            f'&nbsp;&nbsp;속도&nbsp;<span style="color:#388bfd;font-weight:600">{speed.value}</span>step/초</span>'
        ))
        solara.SliderInt("", value=speed.value, min=1, max=30, on_value=speed.set)

    # ── 지도 + 패널 ───────────────────────────────────────────────
    with solara.Row(style="flex:1;overflow:hidden;gap:0;min-height:0;"):
        with solara.Column(style="flex:1;overflow:hidden;min-width:0;"):
            solara.display(map_obj)

        solara.HTML("div", unsafe_innerHTML=_panel_html(
            ag=ag0, step=step_count.value, houses=model.houses,
            walk_km=live_walk_km.value, truck_km=live_truck_km.value,
            carried_kg=live_carried_kg.value, steep=live_steep.value,
            max_hours=max_hours, log_path=log_path.value, scenario=scenario,
            total_orders=total_orders.value, num_workers=num_workers.value,
            steep_km=live_steep_km.value, alley_pct=live_alley_pct.value,
            agents=model.delivery_agents,
        ))


page = Page()
page  # noqa