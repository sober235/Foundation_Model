// anatobind/level_r/app/app.js — Level R reader/adjudicator page (spec §5–§6). Vanilla JS, no build step.
"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";
const ZH = {"white_matter": "白质", "cortex": "皮层", "thalamus": "丘脑", "basal_ganglia": "基底节", "brainstem": "脑干", "cerebellum": "小脑", "other": "其他",
  "periventricular": "脑室旁", "juxtacortical": "近皮层", "cortical": "皮层内", "deep_white_matter": "深部白质", "infratentorial": "幕下",
  "adjacent_to_cortex": "邻近皮层", "adjacent_to_ventricle": "邻近脑室", "crosses_boundary": "跨越边界", "none": "无",
  "certain": "确定", "two_host": "两个宿主难分", "multi_structure": "多结构", "insufficient_resolution": "分辨率不足",
  "good": "好", "fair": "一般", "poor": "差", "not_a_lesion": "不是病灶",
  "nonspecific_wm_lesion": "非特异性白质病灶", "lacunar_infarct": "腔隙性梗死", "perivascular_space": "血管周围间隙",
  "image_left": "图像左侧", "image_right": "图像右侧", "midline": "中线",
  "frontal": "额叶", "parietal": "顶叶", "temporal": "颞叶", "occipital": "枕叶", "insular": "岛叶", "not_applicable": "不适用（不属于任何脑叶：胼胝体、深部灰质、幕下等）"};
const FIELD_ZH = {"lesion_type": "病灶类型", "side": "侧别", "lobe": "脑叶", "primary_host": "主宿主", "acceptable_hosts": "可接受集合",
  "topography": "拓扑位置", "adjacency": "邻接", "ambiguity": "不确定性", "not_a_lesion": "不是病灶", "comment": "备注"};

const state = {me: null, enums: null, mode: "reader", lesion: null, vol: null, z: 0, zoom: 2, win: null, opened: 0, center: null};
const volCache = new Map();
const $ = (id) => document.getElementById(id);

// ---- API --------------------------------------------------------------------------------------------------------
async function api(path, opts) {
  const sep = path.includes("?") ? "&" : "?";
  const r = await fetch(path + sep + "token=" + encodeURIComponent(TOKEN), opts);
  if (r.status === 403) { $("status").textContent = "令牌无效或无权限，请检查链接"; throw new Error("403"); }
  return r;
}
async function apiJSON(path, opts) {
  const r = await api(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || String(r.status));
  return j;
}
async function loadVolume(code) {
  if (volCache.has(code)) return volCache.get(code);
  const meta = await apiJSON(`/api/volume/${code}.json`);
  const r = await api(`/api/volume/${code}.u16`);
  if (!r.ok) throw new Error(`图像加载失败（${r.status}）`);
  const v = {meta, data: new Uint16Array(await r.arrayBuffer())};
  volCache.set(code, v);
  return v;
}

// ---- rendering --------------------------------------------------------------------------------------------------
function sliceImage(v, z, lo, hi) {
  const [S, R, C] = v.meta.shape, img = new ImageData(C, R), d = img.data, base = z * R * C, span = Math.max(1, hi - lo);
  for (let i = 0; i < R * C; i++) {
    let g = (v.data[base + i] - lo) * 255 / span;
    g = g < 0 ? 0 : g > 255 ? 255 : g;
    const o = i * 4; d[o] = d[o + 1] = d[o + 2] = g; d[o + 3] = 255;
  }
  return img;
}
function boxesOn(z) { return state.lesion.boxes[String(z)] || []; }
function lesionCenter(boxes, R, C) {          // mean of the box centres over all slices: one zoom centre per lesion
  const bs = Object.values(boxes).flat();
  if (!bs.length) return [R / 2, C / 2];
  let r = 0, c = 0;
  for (const [r0, r1, c0, c1] of bs) { r += (r0 + r1) / 2; c += (c0 + c1) / 2; }
  return [r / bs.length, c / bs.length];
}
function draw() {
  if (!state.vol) return;
  const v = state.vol, [S, R, C] = v.meta.shape, z = state.z, [lo, hi] = state.win;
  const off = $("off"); off.width = C; off.height = R;
  off.getContext("2d").putImageData(sliceImage(v, z, lo, hi), 0, 0);
  const cv = $("view"), ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  const k = state.zoom, sw = C / k, sh = R / k, [cy, cx] = state.center;
  const ox = k === 1 ? 0 : Math.min(Math.max(0, cx - sw / 2), C - sw), oy = k === 1 ? 0 : Math.min(Math.max(0, cy - sh / 2), R - sh);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(off, ox, oy, sw, sh, 0, 0, cv.width, cv.height);
  const fx = cv.width / sw, fy = cv.height / sh;
  ctx.strokeStyle = "#00e676"; ctx.lineWidth = 2;
  for (const [r0, r1, c0, c1] of boxesOn(z)) ctx.strokeRect((c0 - ox) * fx, (r0 - oy) * fy, (c1 - c0) * fx, (r1 - r0) * fy);
  $("zlabel").textContent = `层 ${z + 1} / ${S}` + (z >= state.lesion.z0 && z <= state.lesion.z1 ? "（病灶层）" : "");
  $("wlabel").textContent = `窗 [${lo}, ${hi}]`;
  $("zslider").value = z;
  thumb("prev", z - 1); thumb("next", z + 1);
}
function thumb(id, z) {
  const cv = $(id), ctx = cv.getContext("2d"), [S, R, C] = state.vol.meta.shape;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (z < 0 || z >= S) return;
  const off = document.createElement("canvas"); off.width = C; off.height = R;
  off.getContext("2d").putImageData(sliceImage(state.vol, z, state.win[0], state.win[1]), 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, cv.width, cv.height);
  ctx.strokeStyle = "#00e676"; ctx.lineWidth = 1;
  const fx = cv.width / C, fy = cv.height / R;
  for (const [r0, r1, c0, c1] of boxesOn(z)) ctx.strokeRect(c0 * fx, r0 * fy, (c1 - c0) * fx, (r1 - r0) * fy);
}
function setZ(z) { const S = state.vol.meta.shape[0]; state.z = Math.min(Math.max(0, z), S - 1); draw(); }

// ---- viewer controls ---------------------------------------------------------------------------------------------
window.addEventListener("keydown", (e) => {
  if (!state.vol || ["TEXTAREA", "SELECT", "INPUT", "BUTTON"].includes(e.target.tagName)) return;
  if (e.key === "ArrowUp" || e.key === "ArrowRight") { e.preventDefault(); setZ(state.z + 1); }
  else if (e.key === "ArrowDown" || e.key === "ArrowLeft") { e.preventDefault(); setZ(state.z - 1); }
});
$("zslider").oninput = (e) => setZ(+e.target.value);
for (const k of [1, 2, 4]) $("zoom" + k).onclick = () => { state.zoom = k; draw(); };
$("wreset").onclick = () => { state.win = [...state.vol.meta.window]; draw(); };
let drag = null;
$("view").addEventListener("mousedown", (e) => { drag = {x: e.clientX, y: e.clientY, win: [...state.win]}; });
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const [lo, hi] = drag.win, width = hi - lo, level = (hi + lo) / 2;
  const nw = Math.max(16, width + (e.clientX - drag.x) * width / 200), nl = level - (e.clientY - drag.y) * width / 200;
  state.win = [Math.round(nl - nw / 2), Math.round(nl + nw / 2)];
  draw();
});
window.addEventListener("mouseup", () => { drag = null; });

// ---- form -------------------------------------------------------------------------------------------------------
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;"); }
function fill(id, keys) { $(id).innerHTML = '<option value="">请选择</option>' + keys.map(k => `<option value="${esc(k)}">${esc(ZH[k] || k)}</option>`).join(""); }
function checkboxes(id, keys) { $(id).innerHTML = keys.map(k => `<label><input type="checkbox" name="${id}" value="${esc(k)}"> ${esc(ZH[k] || k)}</label>`).join(""); }
function radios(id, vals) { $(id).innerHTML = vals.map(v => `<label><input type="radio" name="${id}" value="${v}"> ${v}</label>`).join(""); }
function checked(name) { return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(i => i.value); }
function setChecked(name, vals) { document.querySelectorAll(`input[name="${name}"]`).forEach(i => { i.checked = vals.includes(i.value); }); }
function buildForm(en) {
  fill("lesion_type", en.lesion_types); fill("side", en.sides); fill("lobe", en.lobes);
  fill("primary_host", en.primary_hosts); checkboxes("acceptable_hosts", en.primary_hosts);
  fill("topography", en.topography); checkboxes("adjacency", en.adjacency); fill("ambiguity", en.ambiguity);
  fill("local_quality", en.local_quality); radios("confidence", [1, 2, 3, 4, 5]);
}
function toggleNal() {
  const nal = $("not_a_lesion").checked;
  for (const id of ["lesion_type", "side", "lobe", "primary_host", "topography", "ambiguity"]) $(id).disabled = nal;
  document.querySelectorAll('input[name="acceptable_hosts"]').forEach(i => { i.disabled = nal; });
}
$("not_a_lesion").onchange = toggleNal;
$("primary_host").onchange = () => {
  const v = $("primary_host").value;
  document.querySelectorAll('input[name="acceptable_hosts"]').forEach(i => { if (i.value === v) i.checked = true; });
};
function resetForm() { $("form").reset(); toggleNal(); $("readers").innerHTML = ""; }
function fillForm(a) {
  $("not_a_lesion").checked = !!a.not_a_lesion; toggleNal();
  $("lesion_type").value = a.lesion_type || ""; $("side").value = a.side || ""; $("lobe").value = a.lobe || "";
  $("primary_host").value = a.primary_host || ""; setChecked("acceptable_hosts", a.acceptable_hosts || []);
  $("topography").value = a.topography || ""; setChecked("adjacency", a.adjacency || []); $("ambiguity").value = a.ambiguity || "";
  $("local_quality").value = a.local_quality || ""; setChecked("confidence", a.confidence != null ? [String(a.confidence)] : []);
  $("comment").value = a.comment || ""; $("reason").value = a.reason || "";
}
function readForm() {
  const nal = $("not_a_lesion").checked, adj = state.mode === "adjudicator";
  const p = {lesion_id: state.lesion.lesion_id, not_a_lesion: nal,
    lesion_type: nal ? null : ($("lesion_type").value || null), side: nal ? null : ($("side").value || null),
    lobe: nal ? null : ($("lobe").value || null),
    primary_host: nal ? null : ($("primary_host").value || null), acceptable_hosts: nal ? [] : checked("acceptable_hosts"),
    topography: nal ? null : ($("topography").value || null), adjacency: checked("adjacency"), ambiguity: nal ? null : ($("ambiguity").value || null),
    comment: $("comment").value, time_seconds: (Date.now() - state.opened) / 1000, window: state.win};
  if (adj) p.reason = $("reason").value; else { p.local_quality = $("local_quality").value || null; p.confidence = parseInt(checked("confidence")[0] || "0", 10); }
  const err = validate(p, adj);
  if (err) throw new Error(err);
  return p;
}
function validate(p, adj) {
  const M = state.enums.max_acceptable;
  if (!p.not_a_lesion) {
    if (!p.lesion_type) return "请选择病灶类型";
    if (!p.side) return "请选择侧别";
    if (!p.lobe) return "请选择脑叶";
    if (!p.primary_host) return "请选择主宿主结构";
    if (!p.acceptable_hosts.includes(p.primary_host)) return "可接受集合必须包含主宿主";
    if (!p.topography) return "请选择拓扑位置";
    if (!p.ambiguity) return "请选择不确定性";
  }
  if (p.acceptable_hosts.length > M) return `可接受集合最多 ${M} 个`;
  if (p.adjacency.includes("none") && p.adjacency.length > 1) return "邻接选了“无”就不能再选其他";
  if (adj) { if (!p.reason.trim()) return "请填写裁定理由"; return null; }
  if (!p.local_quality) return "请选择局部图像质量";
  if (!(p.confidence >= 1 && p.confidence <= 5)) return "请选择信心 1–5";
  return null;
}
function fmt(v) {
  if (Array.isArray(v)) return v.length ? v.map(x => ZH[x] || x).join("、") : "—";
  if (v === true) return "是"; if (v === false || v == null || v === "") return "—";
  return ZH[v] || v;
}
function showReaders(views) {
  const keys = ["lesion_type", "side", "lobe", "primary_host", "acceptable_hosts", "topography", "adjacency", "ambiguity", "not_a_lesion", "comment"];
  $("readers").innerHTML = "<table><tr><th></th><th>读者 1</th><th>读者 2</th></tr>" +
    keys.map(k => `<tr><td>${FIELD_ZH[k]}</td>${views.map(v => `<td>${esc(fmt(v[k]))}</td>`).join("")}</tr>`).join("") + "</table>";
}

// ---- navigation -------------------------------------------------------------------------------------------------
function show(name) { $("sec-list").classList.toggle("hidden", name !== "list"); $("sec-lesion").classList.toggle("hidden", name !== "lesion"); }
async function refreshMe() {
  const me = await apiJSON("/api/me");
  state.me = me; state.mode = me.role;
  document.body.classList.toggle("adjudicator", me.role === "adjudicator");
  $("who").textContent = me.display;
  $("progress").textContent = me.role === "reader" ? `已完成 ${me.done} / ${me.total}` : `待裁定 ${me.disagreements}`;
  return me;
}
async function openLesion(lid) {
  const j = await apiJSON(state.mode === "adjudicator" ? `/api/adjudicate/${lid}` : `/api/lesion/${lid}`);
  const vol = await loadVolume(j.lesion.volume_code);
  const [S, R, C] = vol.meta.shape;
  state.lesion = j.lesion;
  state.vol = vol;
  state.win = [...vol.meta.window];
  state.z = Math.floor((j.lesion.z0 + j.lesion.z1) / 2);
  state.zoom = 2; state.opened = Date.now();
  state.center = lesionCenter(j.lesion.boxes, R, C);
  const cv = $("view");
  cv.width = C * 2; cv.height = R * 2; $("zslider").max = S - 1;
  for (const id of ["prev", "next"]) { $(id).width = 200; $(id).height = Math.round(200 * R / C); }
  $("lcode").textContent = `病灶 ${j.lesion.code}`;
  resetForm();
  if (j.answer) fillForm(j.answer);
  if (state.mode === "adjudicator") showReaders(j.readers);
  $("status").textContent = "";
  show("lesion"); draw();
}
async function showList() {
  const rows = await apiJSON(state.mode === "adjudicator" ? "/api/disagreements" : "/api/list");
  $("list").innerHTML = rows.map(r => `<li class="${r.done ? "done" : ""}"><a href="#" data-lid="${esc(r.lesion_id)}">` +
    (state.mode === "adjudicator" ? `病灶 #${esc(r.lesion_id)}` : `第 ${esc(r.position + 1)} 例${r.is_pilot ? "（pilot）" : ""}`) + (r.done ? " ✓" : "") + "</a></li>").join("");
  $("list").querySelectorAll("a").forEach(a => { a.onclick = async (e) => {
    e.preventDefault();
    try { await openLesion(+a.dataset.lid); } catch (err) { $("status").textContent = "加载失败：" + err.message; }
  }; });
  show("list");
}
async function openNext() {
  const me = await refreshMe();
  if (state.mode === "adjudicator") {
    const rows = await apiJSON("/api/disagreements"), n = rows.find(r => !r.done);
    return n ? await openLesion(n.lesion_id) : await showList();
  }
  if (me.held && me.next == null) {            // end of the pilot: wait for the team's release before reading on
    await showList();
    $("status").textContent = "pilot 已完成，请等待通知再继续";
    return;
  }
  return me.next != null ? await openLesion(me.next) : await showList();
}
$("btn-list").onclick = (e) => { e.preventDefault(); showList(); };
$("btn-next").onclick = async (e) => {
  e.preventDefault();
  try { await openNext(); } catch (err) { $("status").textContent = "加载失败：" + err.message; }
};
$("submit").onclick = async () => {
  try {
    const p = readForm();
    await apiJSON(state.mode === "adjudicator" ? "/api/adjudication" : "/api/label",
      {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(p)});
    $("status").textContent = "已保存";
  } catch (e) { $("status").textContent = "未保存：" + e.message; return; }
  try { await openNext(); } catch (e) { $("status").textContent = "已保存，但加载下一例失败：" + e.message; }
};

async function main() {
  state.enums = await apiJSON("/api/enums");
  buildForm(state.enums);
  try { await openNext(); } catch (e) { $("status").textContent = "加载失败：" + e.message; }
}
main().catch(e => { $("status").textContent = "加载失败：" + e.message; });
