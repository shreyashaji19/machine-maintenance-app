const SENSORS = [
  ["air_temp", "Air temperature (K)", 295, 305, 0.1, 298.1],
  ["process_temp", "Process temperature (K)", 305, 314, 0.1, 308.6],
  ["rpm", "Rotational speed (rpm)", 1100, 2900, 10, 1551],
  ["torque", "Torque (Nm)", 3, 77, 0.1, 42.8],
  ["tool_wear", "Tool wear (min)", 0, 255, 1, 0],
];
const PRESETS = {
  "Healthy run": { air_temp: 298.1, process_temp: 308.6, rpm: 1551, torque: 42.8, tool_wear: 0, type: "M" },
  "Worn tool under load": { air_temp: 300, process_temp: 309, rpm: 1300, torque: 65, tool_wear: 220, type: "L" },
  "Heat stress": { air_temp: 302, process_temp: 309.5, rpm: 1300, torque: 45, tool_wear: 60, type: "L" },
};
const $ = (s) => document.querySelector(s);
const state = { type: "L" };
const MESSAGES = {
  normal: ["Normal operation", "Readings are in a safe range. No action needed."],
  elevated: ["Elevated risk", "Readings are drifting. Plan an inspection at the next stop."],
  failure: ["Failure likely", "Schedule maintenance now to avoid unplanned downtime."],
};

// Build sliders
$("#sliders").innerHTML = SENSORS.map(([k, label, min, max, step, val]) =>
  `<div class="field"><div class="lab"><label for="${k}">${label}</label><output id="o_${k}">${val}</output></div>
   <input type="range" id="${k}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`).join("");
// Gauge ticks
$("#ticks").innerHTML = Array.from({ length: 11 }, (_, i) => {
  const a = (-180 + i * 18) * Math.PI / 180, r1 = 132, r2 = i % 5 ? 138 : 144;
  return `<line x1="${150 + r1 * Math.cos(a)}" y1="${150 + r1 * Math.sin(a)}" x2="${150 + r2 * Math.cos(a)}" y2="${150 + r2 * Math.sin(a)}"/>`;
}).join("");
$("#presets").innerHTML = Object.keys(PRESETS).map((n) => `<button>${n}</button>`).join("");

const readings = () => ({ ...Object.fromEntries(SENSORS.map(([k]) => [k, +$("#" + k).value])), type: state.type });

function setType(t) {
  state.type = t;
  document.querySelectorAll("#type button").forEach((b) => b.classList.toggle("on", b.dataset.v === t));
}
function fill(values) {
  SENSORS.forEach(([k]) => { $("#" + k).value = values[k]; $("#o_" + k).textContent = values[k]; });
  setType(values.type); score();
}

let timer;
function score() { clearTimeout(timer); timer = setTimeout(run, 150); }
async function run() {
  try {
    const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(readings()) });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error);
    document.body.dataset.state = d.status;
    $("#needle").style.transform = `rotate(${d.probability * 180 - 90}deg)`;
    $("#pct").textContent = (d.probability * 100).toFixed(1);
    [$("#verdict").textContent, $("#hint").textContent] = MESSAGES[d.status];
  } catch (e) {
    $("#verdict").textContent = "Model unavailable";
    $("#hint").textContent = e.message || "Run python -m src.train, then restart the app.";
  }
}

SENSORS.forEach(([k]) => $("#" + k).addEventListener("input", (e) => { $("#o_" + k).textContent = e.target.value; score(); }));
document.querySelectorAll("#type button").forEach((b) => b.addEventListener("click", () => { setType(b.dataset.v); score(); }));
document.querySelectorAll("#presets button").forEach((b) => b.addEventListener("click", () => fill(PRESETS[b.textContent])));

const bars = (el, rows, best) => $(el).innerHTML = rows.map(([n, v]) =>
  `<div class="bar ${n === best ? "best" : ""}"><span>${n}${n === best ? " ★" : ""}</span><i style="width:${v * 100}%"></i><b>${v.toFixed(2)}</b></div>`).join("");
fetch("/api/metrics").then((r) => r.json()).then((m) => {
  if (m.error) return;
  bars("#models", Object.entries(m.models).map(([n, s]) => [n, s.f1]), m.best_model);
  const imp = Object.entries(m.importances).sort((a, b) => b[1] - a[1]);
  bars("#imps", imp.map(([n, v]) => [n.replace(/ \[.*\]/, ""), v * 3]));
});
fill(PRESETS["Healthy run"]);
