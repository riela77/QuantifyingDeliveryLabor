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
from pathlib import Path
import solara
from model import DeliveryModel, NUM_HOUSES, START_LAT, START_LON
from agents import TruckWalkAgent, MotoAgent, STEEP_THRESHOLD

# ── CSS 문자열 (JS로 document.head에 주입) ──────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:      #0d1117;
  --surf:    #161b22;
  --overlay: #21262d;
  --bd:      #30363d;
  --bd-dim:  #21262d;
  --t1:      #e6edf3;
  --t2:      #c9d1d9;
  --t3:      #8b949e;
  --t4:      #484f58;
  --blue:    #1f6feb;
  --blue-hi: #388bfd;
  --grn-hi:  #2ea043;
  --yel-hi:  #d29922;
  --red:     #da3633;
  --red-hi:  #f85149;
  --org-hi:  #f0883e;
  --mono:    'JetBrains Mono', monospace;
}

/* ── 전체 페이지 다크 배경 ── */
html, body { background: var(--bg) !important; }
.v-application { background: var(--bg) !important; }
.v-application--wrap { background: var(--bg) !important; }
.v-main, .v-main__wrap { background: var(--bg) !important; }
.v-content, .v-content__wrap { background: var(--bg) !important; }

/* Solara 기본 카드/시트 투명 처리 */
.v-card, .v-sheet { background: transparent !important; box-shadow: none !important; }

/* ── 헤더 ── */
.lm-header {
  background: var(--surf) !important;
  border-bottom: 1px solid var(--bd);
  height: 44px; display: flex; align-items: center;
  padding: 0 16px; gap: 10px; width: 100%;
}
.lm-logo { font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--t1); display: flex; align-items: center; gap: 5px; }
.lm-logo .br { color: var(--blue-hi); }
.lm-logo-sep { width: 1px; height: 18px; background: var(--bd); }
.lm-logo-sub { font-size: 10px; color: var(--t3); font-family: var(--mono); }
.lm-ticker {
  margin-left: auto; display: flex; align-items: center; gap: 7px;
  background: var(--overlay); border: 0.5px solid var(--bd);
  border-radius: 6px; padding: 3px 10px;
  font-family: var(--mono); font-size: 10px; color: var(--t3);
}
.lm-ticker .sv { font-size: 12px; font-weight: 600; color: var(--t1); }
.lm-ticker .ph {
  font-size: 9px; padding: 1px 6px; border-radius: 4px;
  background: var(--bg); color: var(--blue-hi);
  border: 0.5px solid var(--blue);
}

/* ── 컨트롤 바 ── */
.lm-ctrl-bar {
  background: var(--surf) !important;
  border-bottom: 1px solid var(--bd);
  padding: 7px 16px; display: flex;
  align-items: center; gap: 8px; flex-wrap: wrap;
}

/* Solara 버튼 → 다크 스타일 강제 */
.lm-ctrl-bar .v-btn {
  background: var(--overlay) !important;
  border: 0.5px solid var(--bd) !important;
  border-radius: 7px !important;
  color: var(--t2) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  box-shadow: none !important;
  height: 32px !important;
  min-height: 32px !important;
}
.lm-ctrl-bar .v-btn:hover { background: var(--bd) !important; color: var(--t1) !important; }
.lm-ctrl-bar .v-btn.btn-run { background: var(--blue) !important; border-color: var(--blue) !important; color: #fff !important; }
.lm-ctrl-bar .v-btn.btn-run:hover { background: var(--blue-hi) !important; }
.lm-ctrl-bar .v-btn.btn-pause { background: var(--red) !important; border-color: var(--red) !important; color: #fff !important; }

/* 슬라이더 다크 */
.lm-ctrl-bar .v-slider { margin: 0 !important; }
.lm-ctrl-bar .v-slider__track-fill { background: var(--blue) !important; }
.lm-ctrl-bar .v-slider__thumb-container .v-slider__thumb { background: var(--blue-hi) !important; }
.lm-ctrl-bar .v-slider__track-background { background: var(--overlay) !important; }
.lm-ctrl-bar .v-input, .lm-ctrl-bar .v-input__slot { background: transparent !important; }
.lm-ctrl-bar label, .lm-ctrl-bar .v-label { color: var(--t3) !important; font-family: var(--mono) !important; font-size: 11px !important; }

/* ── 지도 위 오버레이 ── */
.lm-status-bar {
  position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%);
  background: rgba(22,27,34,.92); border: 0.5px solid var(--bd);
  border-radius: 100px; padding: 5px 18px;
  font-size: 11px; font-family: var(--mono); color: var(--t3);
  white-space: nowrap; pointer-events: none; z-index: 500;
  backdrop-filter: blur(8px);
}
.lm-map-alert {
  position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
  background: rgba(13,10,10,.95); border: 0.5px solid var(--red-hi);
  border-radius: 8px; padding: 7px 16px;
  font-size: 11px; font-family: var(--mono); color: var(--red-hi);
  white-space: nowrap; z-index: 600; pointer-events: none;
}
.lm-legend {
  position: absolute; bottom: 18px; right: 14px;
  background: rgba(22,27,34,.9); border: 0.5px solid var(--bd);
  border-radius: 8px; padding: 9px 12px;
  font-size: 10px; color: var(--t3); font-family: var(--mono); z-index: 500;
  line-height: 2.1;
}
.lm-leg { display: flex; align-items: center; gap: 6px; }
.lm-ll  { display: inline-block; width: 18px; height: 2.5px; border-radius: 2px; }
.lm-ldot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

/* ── 우측 패널 ── */
.lm-panel {
  width: 272px !important; flex-shrink: 0 !important;
  background: var(--surf) !important;
  border-left: 1px solid var(--bd) !important;
  display: flex !important; flex-direction: column;
  overflow: hidden; height: 100%;
}
.lm-tabs { display: flex; border-bottom: 1px solid var(--bd); flex-shrink: 0; }
.lm-tab {
  flex: 1; padding: 9px 4px; font-size: 10px; font-family: var(--mono);
  color: var(--t3); cursor: pointer; border: none;
  border-bottom: 2px solid transparent; background: none; letter-spacing: .02em;
  transition: all .15s;
}
.lm-tab:hover { color: var(--t2); background: var(--overlay); }
.lm-tab.active { color: var(--blue-hi); border-bottom-color: var(--blue); }
.lm-pane { display: none; flex: 1; overflow-y: auto; padding: 14px; flex-direction: column; gap: 9px; }
.lm-pane.active { display: flex; }
.lm-pane::-webkit-scrollbar { width: 3px; }
.lm-pane::-webkit-scrollbar-thumb { background: var(--bd); border-radius: 2px; }

/* ── 지표 카드 ── */
.lm-mgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.lm-metric { background: var(--bg); border: 0.5px solid var(--bd); border-radius: 8px; padding: 10px 11px; }
.lm-mv { font-size: 22px; font-weight: 600; font-family: var(--mono); line-height: 1; }
.lm-ml { font-size: 10px; color: var(--t3); margin-top: 4px; }

/* ── 섹션 라벨 ── */
.lm-sec { font-size: 10px; color: var(--t3); font-family: var(--mono); font-weight: 500; letter-spacing: .06em; text-transform: uppercase; border-bottom: 0.5px solid var(--bd-dim); padding-bottom: 4px; margin-top: 2px; }

/* ── 진행바 ── */
.lm-prow { display: flex; justify-content: space-between; font-size: 10px; color: var(--t3); font-family: var(--mono); margin-bottom: 5px; }
.lm-pwrap { background: var(--overlay); border-radius: 4px; height: 5px; overflow: hidden; }
.lm-pfill { height: 5px; border-radius: 4px; background: var(--blue-hi); transition: width .4s; }

/* ── 노동 강도 카드 ── */
.lm-lgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.lm-lc { background: var(--bg); border: 0.5px solid var(--bd); border-radius: 8px; padding: 9px 10px; }
.lm-lc-lbl { font-size: 9px; color: var(--t3); font-family: var(--mono); letter-spacing: .04em; text-transform: uppercase; margin-bottom: 5px; }
.lm-lc-val { font-size: 17px; font-weight: 600; font-family: var(--mono); color: var(--t1); }
.lm-lc-val.warn   { color: var(--org-hi); }
.lm-lc-val.danger { color: var(--red-hi); }

/* ── 경사도 박스 ── */
.lm-slopebox { background: var(--bg); border: 0.5px solid var(--bd); border-radius: 8px; padding: 9px 12px; }
.lm-sloperow { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }

/* ── 배달원 카드 ── */
.lm-drv { background: var(--bg); border: 0.5px solid var(--bd); border-radius: 8px; padding: 9px 11px; transition: border-color .25s, background .25s; }
.lm-drv.warn { border-color: var(--yel-hi) !important; background: rgba(18,16,10,.6); }
.lm-drv.over { border-color: var(--red-hi) !important; background: rgba(19,10,10,.6); }
.lm-drv-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.lm-drv-name { font-family: var(--mono); font-weight: 600; font-size: 11px; color: var(--t1); }
.lm-badge { font-size: 9px; font-family: var(--mono); padding: 2px 7px; border-radius: 20px; background: var(--overlay); color: var(--t3); border: 0.5px solid var(--bd); }
.lm-badge.truck { color: var(--blue-hi); border-color: var(--blue); }
.lm-badge.walk  { color: var(--org-hi);  border-color: var(--org-hi); }
.lm-badge.done  { color: var(--grn-hi);  border-color: var(--grn-hi); }
.lm-badge.over  { color: var(--red-hi);  border-color: var(--red-hi); }
.lm-drow { display: flex; justify-content: space-between; color: var(--t3); font-size: 10px; font-family: var(--mono); margin-bottom: 3px; }
.lm-drow span:last-child { color: var(--t2); }
.lm-dpbg { background: var(--overlay); border-radius: 3px; height: 3px; margin-top: 5px; }
.lm-dpfg { border-radius: 3px; height: 3px; background: var(--blue); transition: width .4s; }
.lm-dpfg.warn { background: var(--yel-hi); }
.lm-dpfg.over { background: var(--red-hi); }

/* ── 배송지 목록 ── */
.lm-hlist { display: flex; flex-direction: column; gap: 3px; }
.lm-hrow { display: flex; align-items: center; gap: 7px; font-size: 11px; font-family: var(--mono); color: var(--t3); padding: 3px 0; }
.lm-hrow.cur  { color: var(--t1); }
.lm-hrow.done { color: var(--t4); }
.lm-hdot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

/* ── 완료 배너 ── */
.lm-done { background: rgba(35,134,54,.1); border: 0.5px solid var(--grn-hi); border-radius: 8px; padding: 10px 12px; font-family: var(--mono); font-size: 11px; color: var(--grn-hi); text-align: center; line-height: 1.9; }

/* ── 시간초과 알림 ── */
.lm-ot { background: rgba(19,10,10,.8); border: 0.5px solid var(--red-hi); border-radius: 6px; padding: 7px 10px; font-size: 11px; color: var(--red-hi); font-family: var(--mono); }

/* ── 인포 박스 ── */
.lm-info { background: var(--bg); border: 0.5px solid var(--bd); border-radius: 8px; padding: 10px 12px; font-size: 10px; font-family: var(--mono); color: var(--t3); line-height: 2.1; }
.lm-info .g { color: var(--grn-hi); font-weight: 500; }
.lm-info .y { color: var(--yel-hi); font-weight: 500; }
.lm-info .r { color: var(--red-hi); font-weight: 500; }
"""

# JS로 document.head에 <style> 태그 직접 박기
_CSS_INJECT_JS = f"""
<script>
(function() {{
  if (document.getElementById('lm-global-css')) return;
  var s = document.createElement('style');
  s.id = 'lm-global-css';
  s.textContent = {repr(_CSS)};
  document.head.appendChild(s);
  // 폰트도 주입
  var f = document.createElement('link');
  f.rel = 'stylesheet';
  f.href = 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap';
  document.head.appendChild(f);
}})();
</script>
"""

# 탭 전환 JS
_TAB_JS = """
<script>
(function() {
  function lmTab(id, btn) {
    document.querySelectorAll('.lm-pane').forEach(function(p){ p.classList.remove('active'); });
    document.querySelectorAll('.lm-tab').forEach(function(b){ b.classList.remove('active'); });
    var pane = document.getElementById('lm-pane-' + id);
    if (pane) pane.classList.add('active');
    if (btn) btn.classList.add('active');
  }
  window.lmTab = lmTab;

  var _alertTimer = null;
  function lmAlert(msg) {
    var el = document.querySelector('.lm-map-alert');
    if (!el) return;
    el.querySelector('span').textContent = msg;
    el.style.display = 'block';
    if (_alertTimer) clearTimeout(_alertTimer);
    _alertTimer = setTimeout(function(){ el.style.display = 'none'; }, 3500);
  }
  window.lmAlert = lmAlert;
})();
</script>
"""


# ── 우측 패널 HTML 생성 함수 ────────────────────────────────────
def _panel_html(ag, step, houses, walk_km, truck_km,
                carried_kg, steep, max_hours, log_path, scenario):
    is_truck  = isinstance(ag, TruckWalkAgent)
    phase     = ag.phase
    delivered = ag.delivered_count
    slope     = ag.current_slope
    over_time = step > max_hours * 60
    warn_time = step > max_hours * 60 * 0.8
    pct       = int(delivered / max(len(houses), 1) * 100)
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

    hlist = ""
    for i, h in enumerate(houses):
        if h.visited:
            dot="2ea043"; cls="lm-hrow done"; icon="✓"
        elif i == ag.house_idx and phase not in ("done","idle"):
            dot="388bfd"; cls="lm-hrow cur";  icon="▶"
        else:
            dot="484f58"; cls="lm-hrow";      icon="○"
        hlist += (f'<div class="{cls}">'
                  f'<span class="lm-hdot" style="background:#{dot}"></span>'
                  f'<span>{icon} 배송지 {i+1}</span>'
                  f'<span style="margin-left:auto;color:var(--t4)">{h.pkg_kg}kg</span>'
                  f'</div>')

    truck_card = (
        f'<div class="lm-lc"><div class="lm-lc-lbl">트럭 이동</div>'
        f'<div class="lm-lc-val">{truck_km:.2f} km</div></div>'
        if is_truck else ""
    )
    log_line = (
        f'<div style="font-size:9px;color:var(--grn-hi);font-family:var(--mono);margin-top:4px">💾 {log_path}</div>'
        if log_path else ""
    )
    done_block = (
        f'<div class="lm-done">🎉 모든 배송 완료<br>'
        f'<span style="font-size:9px;color:var(--t3)">총 {step}분 · 도보 {walk_km:.2f}km · 험지 {steep}회</span>'
        f'{log_line}</div>'
    ) if phase == "done" else ""

    ot_block = (
        '<div class="lm-ot">⛔ 근무시간 초과</div>'
    ) if over_time else ""

    return f"""
<div class="lm-panel">
  <div class="lm-tabs">
    <button class="lm-tab active" onclick="lmTab('status',this)">📊 현황</button>
    <button class="lm-tab"       onclick="lmTab('params',this)">⚙️ 파라미터</button>
    <button class="lm-tab"       onclick="lmTab('env',this)">🗺️ 환경</button>
  </div>

  <div class="lm-pane active" id="lm-pane-status">
    <div class="lm-mgrid">
      <div class="lm-metric">
        <div class="lm-mv" style="color:#2ea043">{1 if phase not in ('done','idle') else 0}</div>
        <div class="lm-ml">🚛 가동 중</div>
      </div>
      <div class="lm-metric">
        <div class="lm-mv" style="color:#f85149">{1 if over_time else 0}</div>
        <div class="lm-ml">⛔ 시간초과</div>
      </div>
      <div class="lm-metric">
        <div class="lm-mv" style="color:#388bfd">{delivered}</div>
        <div class="lm-ml">📦 완료</div>
      </div>
      <div class="lm-metric">
        <div class="lm-mv" style="color:#f0883e">{1 if phase=='walk' else 0}</div>
        <div class="lm-ml">🚶 도보 中</div>
      </div>
    </div>

    <div>
      <div class="lm-prow">
        <span>전체 진행률</span>
        <span style="color:#e6edf3">{delivered}/{len(houses)} ({pct}%)</span>
      </div>
      <div class="lm-pwrap"><div class="lm-pfill" style="width:{pct}%"></div></div>
    </div>

    {ot_block}

    <div style="background:#0d1117;border:0.5px solid #30363d;border-radius:8px;
                padding:9px 12px;display:flex;justify-content:space-between;align-items:center">
      <span style="font-family:var(--mono);font-size:10px;color:#8b949e">⏱ 경과</span>
      <span style="font-family:var(--mono);font-size:18px;font-weight:600;color:#e6edf3">{hh:02d}:{mm:02d}</span>
    </div>

    <div class="lm-sec">노동 강도 지표</div>
    <div class="lm-lgrid">
      <div class="lm-lc">
        <div class="lm-lc-lbl">{'누적 도보' if is_truck else '누적 이동'}</div>
        <div class="lm-lc-val warn">{walk_km:.2f} km</div>
      </div>
      {truck_card}
      <div class="lm-lc">
        <div class="lm-lc-lbl">현재 화물</div>
        <div class="lm-lc-val danger">{carried_kg:.1f} kg</div>
      </div>
      <div class="lm-lc">
        <div class="lm-lc-lbl">험지 돌파</div>
        <div class="lm-lc-val danger">{steep} 회</div>
      </div>
    </div>

    <div class="lm-slopebox">
      <div class="lm-sloperow">
        <span style="font-size:10px;color:#8b949e;font-family:var(--mono)">현재 경사도</span>
        <span style="font-family:var(--mono);font-weight:600;font-size:13px;color:#e6edf3">{abs(slope):.1f}°</span>
      </div>
      <div class="lm-pwrap">
        <div class="lm-pfill" style="width:{sl_pct:.1f}%;background:{sl_color};transition:width .5s,background .5s"></div>
      </div>
    </div>

    <div class="lm-sec">배달원 현황</div>
    <div class="{drv_cls}">
      <div class="lm-drv-head">
        <span class="lm-drv-name">#01 배달원</span>
        <span class="{badge_cls}">{badge_txt}</span>
      </div>
      <div class="lm-drow"><span>도보 거리</span><span>{walk_km:.2f} km</span></div>
      <div class="lm-drow"><span>배송 완료</span><span>{delivered}건</span></div>
      <div class="lm-drow"><span>현재 화물</span><span>{carried_kg:.1f} kg</span></div>
      <div class="lm-drow"><span>경과 시간</span><span>{hh:02d}:{mm:02d}</span></div>
      <div class="lm-dpbg"><div class="{bar_cls}" style="width:{pct}%"></div></div>
    </div>

    <div class="lm-sec">배송지 목록</div>
    <div class="lm-hlist">{hlist}</div>
    {done_block}
  </div>

  <div class="lm-pane" id="lm-pane-params">
    <div class="lm-sec">시뮬레이션 정보</div>
    <div class="lm-info">
      배송지: <span style="color:#e6edf3">{len(houses)}곳</span><br>
      총 화물: <span style="color:#e6edf3">{ag.delivered_kg:.1f} kg</span><br>
      예상 시간: <span style="color:#e6edf3">{ag.total_time_min:.0f}분</span><br>
      예상 비용: <span style="color:#e6edf3">{ag.total_cost_won:,.0f}원</span>
    </div>
    <div class="lm-sec" style="margin-top:4px">피로도 모델</div>
    <div class="lm-info">
      경사 &lt;15° → 기본 속도<br>
      경사 15~25° → <span class="y">×0.5 감속</span><br>
      경사 25°+ &nbsp;→ <span class="r">×0.3 감속</span><br>
      협로 (폭≤4m) → <span class="r">도보 강제전환</span>
    </div>
  </div>

  <div class="lm-pane" id="lm-pane-env">
    <div class="lm-sec">도로 데이터</div>
    <div class="lm-info">
      출발지: <span style="color:#e6edf3">우림시장</span><br>
      좌표계: <span style="color:#e6edf3">EPSG 4326</span><br>
      DEM: <span style="color:#e6edf3">90m → 5m 보간</span>
    </div>
    <div class="lm-sec" style="margin-top:4px">도로 범례</div>
    <div style="font-size:10px;font-family:var(--mono);color:#8b949e;line-height:2.2">
      <div class="lm-leg"><span class="lm-ll" style="background:#238636"></span>평지 &lt;15°</div>
      <div class="lm-leg"><span class="lm-ll" style="background:#d29922"></span>경사 15~25°</div>
      <div class="lm-leg"><span class="lm-ll" style="background:#f85149"></span>급경사 25°+</div>
      <div style="border-top:0.5px solid #21262d;margin:5px 0"></div>
      <div class="lm-leg"><span class="lm-ldot" style="background:#388bfd"></span>트럭 이동</div>
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
    max_hours              = 8

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
            layout={'height': '780px', 'width': '100%'}  # 👈 여기에 layout 속성을 추가하세요!
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

    # ── 지도 초기화 ───────────────────────────────────────────────
    def init_map():
        for key in list(marker_dict.keys()):
            del marker_dict[key]
        for layer in [l for l in map_obj.layers if not isinstance(l, ipl.TileLayer)]:
            map_obj.remove(layer)
        ag = model.delivery_agent
        if isinstance(ag, TruckWalkAgent):
            for seg in ag.segments:
                if seg["truck"]:
                    map_obj.add(ipl.Polyline(
                        locations=[[lat, lon] for lon, lat in seg["truck"]],
                        color="#388bfd", weight=3, opacity=0.65))
                if seg["walk"]:
                    map_obj.add(ipl.Polyline(
                        locations=[[lat, lon] for lon, lat in seg["walk"]],
                        color="#f0883e", weight=2, opacity=0.75, dash_array="6 4"))
        else:
            for seg in ag.segments:
                if seg["route"]:
                    map_obj.add(ipl.Polyline(
                        locations=[[lat, lon] for lon, lat in seg["route"]],
                        color="#f0883e", weight=3, opacity=0.7))

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

        map_obj.add(ipl.Marker(
            location=(START_LAT, START_LON),
            icon=ipl.AwesomeIcon(name="home", marker_color="blue"),
            title="우림시장 (출발지)", draggable=False))
        for i, house in enumerate(model.houses):
            m = ipl.CircleMarker(
                location=(house.geometry.y, house.geometry.x),
                radius=9, color="#484f58", fill_color="#484f58",
                fill_opacity=0.9, weight=2,
                title=f"배송지 {i+1} | {house.pkg_kg}kg")
            marker_dict[f"house_{house.unique_id}"] = m
            map_obj.add(m)
        icon = ipl.AwesomeIcon(
            name="truck" if isinstance(ag, TruckWalkAgent) else "motorcycle",
            marker_color="cadetblue")
        agent_marker = ipl.Marker(
            location=(ag.geometry.y, ag.geometry.x),
            icon=icon, draggable=False, title="배달원")
        marker_dict["delivery_agent"] = agent_marker
        map_obj.add(agent_marker)

    solara.use_effect(init_map, [scenario])

    # ── 시뮬레이션 루프 ───────────────────────────────────────────
    def run_simulation():
        if not running.value:
            return
        ag = model.delivery_agent
        prev_phase = ag.phase
        while running.value:
            if ag.phase == "done":
                if not log_saved.value:
                    path = model.save_logs()
                    log_saved.set(True); log_path.set(path or "")
                    alert_msg.set("💾 로그 저장 완료")
                running.set(False); break
            model.step()
            if "delivery_agent" in marker_dict:
                marker_dict["delivery_agent"].location = (ag.geometry.y, ag.geometry.x)
                if isinstance(ag, TruckWalkAgent):
                    color = "cadetblue" if ag.phase == "truck" else "orange"
                    name  = "truck"     if ag.phase == "truck" else "male"
                    marker_dict["delivery_agent"].icon = ipl.AwesomeIcon(name=name, marker_color=color)
            for house in model.houses:
                key = f"house_{house.unique_id}"
                if key in marker_dict and house.visited:
                    marker_dict[key].color = "#2ea043"
                    marker_dict[key].fill_color = "#2ea043"
            cur = ag.phase
            if cur != prev_phase:
                if cur == "walk":
                    seg = ag.current_segment
                    kg  = seg["house"].pkg_kg if seg else 0
                    alert_msg.set(f"차량 진입 불가 — 도보 전환 ({kg}kg)")
                elif cur == "truck":
                    alert_msg.set("다음 배송지로 트럭 이동 중")
                prev_phase = cur
            elif cur == "walk" and abs(ag.current_slope) >= STEEP_THRESHOLD:
                alert_msg.set(f"급경사 구간 진입 ({ag.current_slope:.1f}°)")
            if isinstance(ag, TruckWalkAgent):
                live_walk_km.set(round(ag.cumul_walk_m  / 1000, 3))
                live_truck_km.set(round(ag.cumul_truck_m / 1000, 3))
            else:
                live_walk_km.set(round(ag.cumul_dist_m  / 1000, 3))
            live_carried_kg.set(ag.carried_kg)
            live_steep.set(ag.steep_crossings)
            step_count.set(step_count.value + 1)
            time.sleep(1 / max(speed.value, 1))

    solara.use_thread(run_simulation, [running.value])

    def on_run():   running.set(not running.value)
    def on_step():
        ag = model.delivery_agent
        if ag.phase != "done":
            model.step()
            if isinstance(ag, TruckWalkAgent):
                live_walk_km.set(round(ag.cumul_walk_m  / 1000, 3))
                live_truck_km.set(round(ag.cumul_truck_m / 1000, 3))
            else:
                live_walk_km.set(round(ag.cumul_dist_m  / 1000, 3))
            live_carried_kg.set(ag.carried_kg)
            live_steep.set(ag.steep_crossings)
            step_count.set(step_count.value + 1)
    def on_reset():
        set_scenario(scenario)
        log_saved.set(False); log_path.set(""); step_count.set(0)
        live_walk_km.set(0.0); live_truck_km.set(0.0)
        live_carried_kg.set(0.0); live_steep.set(0); alert_msg.set("")
    def on_log():
        path = model.save_logs(); log_path.set(path or "")
        alert_msg.set(f"💾 로그 저장: {path}")

    ag = model.delivery_agent
    solara.Title("LastMile Labor")

    # ── CSS를 JS로 document.head에 강제 주입 ─────────────────────
    solara.HTML("div", unsafe_innerHTML=_CSS_INJECT_JS)
    solara.HTML("div", unsafe_innerHTML=_TAB_JS)

    # ── 헤더 ─────────────────────────────────────────────────────
    hh_e, mm_e = step_count.value // 60, step_count.value % 60
    solara.HTML("div", unsafe_innerHTML=f"""
    <div class="lm-header">
      <div class="lm-logo"><span class="br">[</span>LM<span class="br">]</span>&nbsp;LastMile Labor</div>
      <div class="lm-logo-sep"></div>
      <div class="lm-logo-sub">배송 노동 가시화 · ABM · 난곡동</div>
      <div class="lm-ticker">
        <span class="sv">{step_count.value}</span>
        <span style="color:#484f58">step</span>
        <span class="ph">{ag.phase.upper()}</span>
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

    with solara.Row(style=(
        "background:#161b22!important;border-bottom:1px solid #30363d;"
        "padding:7px 16px;gap:8px;flex-wrap:wrap;align-items:center;"
    )):
        solara.Button(run_lbl,      on_click=on_run,   style=run_style + btn_base)
        solara.Button("▷ 1 Step",  on_click=on_step,  style=btn_sec, disabled=running.value)
        solara.Button("↺ 초기화",  on_click=on_reset, style=btn_sec, disabled=running.value)
        solara.Button("💾 로그 저장", on_click=on_log, style=btn_sec)
        solara.HTML("span", unsafe_innerHTML=(
            f'<span style="font-family:monospace;font-size:11px;color:#8b949e;">'
            f'속도&nbsp;<span style="color:#388bfd;font-weight:600">{speed.value}</span>'
            f'&nbsp;step/초</span>'
        ))
        solara.SliderInt("", value=speed.value, min=1, max=30, on_value=speed.set)

    # ── 지도 + 패널 ───────────────────────────────────────────────
    with solara.Row(style="flex:1;overflow:hidden;gap:0;min-height:0;"):

        # 지도 영역
        with solara.Column(style="flex:1;position:relative;overflow:hidden;min-width:0;"):
            solara.display(map_obj)
            alert_disp = "block" if alert_msg.value else "none"
            status_txt = (
                "▶ 실행을 눌러 시작하세요" if step_count.value == 0 else
                "🎉 모든 배송 완료!"        if ag.phase == "done"  else
                f"▶ 실행 중  Step {step_count.value}"
            )
            solara.HTML("div", unsafe_innerHTML=f"""
            <div class="lm-map-alert" style="display:{alert_disp}">
              ⚠ <span>{alert_msg.value}</span>
            </div>
            <div class="lm-status-bar">{status_txt}</div>
            <div class="lm-legend">
              <div class="lm-leg"><span class="lm-ll" style="background:#238636"></span>평지 &lt;15°</div>
              <div class="lm-leg"><span class="lm-ll" style="background:#d29922"></span>경사 15~25°</div>
              <div class="lm-leg"><span class="lm-ll" style="background:#f85149"></span>급경사 25°+</div>
              <div style="border-top:0.5px solid #21262d;margin:4px 0"></div>
              <div class="lm-leg"><span class="lm-ldot" style="background:#388bfd"></span>트럭 이동</div>
              <div class="lm-leg"><span class="lm-ldot" style="background:#f0883e"></span>도보 배송</div>
              <div class="lm-leg"><span class="lm-ldot" style="background:#2ea043"></span>배송 완료</div>
            </div>""")

        # 우측 패널 (순수 HTML)
        solara.HTML("div", unsafe_innerHTML=_panel_html(
            ag=ag, step=step_count.value, houses=model.houses,
            walk_km=live_walk_km.value, truck_km=live_truck_km.value,
            carried_kg=live_carried_kg.value, steep=live_steep.value,
            max_hours=max_hours, log_path=log_path.value, scenario=scenario,
        ))


page = Page()
page  # noqa