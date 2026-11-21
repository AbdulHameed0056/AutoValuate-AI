const INR_PER_USD = 84;
const HISTORY_KEY = "autovaluate_history_v1";
const MAX_HISTORY = 6;
const DEV_FRONTEND_PORTS = new Set(["5500", "5501", "3000", "5173"]);
const API_BASE =
  window.location.protocol === "file:" || DEV_FRONTEND_PORTS.has(window.location.port)
    ? "http://127.0.0.1:5000"
    : "";

const PRESETS = {
  creta: {
    brand: "Hyundai",
    model: "Creta",
    year: 2021,
    km_driven: 42000,
    fuel_type: "Diesel",
    transmission: "Automatic",
    owner: "First Owner",
    seller_type: "Dealer",
    condition: "Excellent",
  },
  swift: {
    brand: "Maruti Suzuki",
    model: "Swift",
    year: 2019,
    km_driven: 56000,
    fuel_type: "Petrol",
    transmission: "Manual",
    owner: "First Owner",
    seller_type: "Individual",
    condition: "Good",
  },
  nexon_ev: {
    brand: "Tata",
    model: "Nexon",
    year: 2023,
    km_driven: 21000,
    fuel_type: "Electric",
    transmission: "Automatic",
    owner: "First Owner",
    seller_type: "Trustmark Dealer",
    condition: "Excellent",
  },
  bmw: {
    brand: "BMW",
    model: "3 Series",
    year: 2020,
    km_driven: 38000,
    fuel_type: "Diesel",
    transmission: "Automatic",
    owner: "Second Owner",
    seller_type: "Trustmark Dealer",
    condition: "Good",
  },
};

const state = {
  catalog: null,
  currency: "INR",
  condition: "Good",
  lastResult: null,
  debounceTimer: null,
};

const dom = {
  form: document.getElementById("prediction-form"),
  brandSelect: document.getElementById("brand-select"),
  modelSelect: document.getElementById("model-select"),
  bodyBadge: document.getElementById("body-type-badge"),
  yearRange: document.getElementById("year-range"),
  yearDisplay: document.getElementById("year-display"),
  carAgeLabel: document.getElementById("car-age-label"),
  minYearLabel: document.getElementById("min-year-label"),
  maxYearLabel: document.getElementById("max-year-label"),
  kmRange: document.getElementById("km-range"),
  kmInput: document.getElementById("km-input"),
  fuelSelect: document.getElementById("fuel-select"),
  transmissionSelect: document.getElementById("transmission-select"),
  ownerSelect: document.getElementById("owner-select"),
  sellerSelect: document.getElementById("seller-select"),
  conditionGroup: document.getElementById("condition-group"),
  engineCc: document.getElementById("engine-cc"),
  powerBhp: document.getElementById("power-bhp"),
  mileageKmpl: document.getElementById("mileage-kmpl"),
  liveToggle: document.getElementById("live-mode-toggle"),
  errorBanner: document.getElementById("error-banner"),
  predictBtn: document.getElementById("predict-btn"),
  modelBadgeText: document.getElementById("model-badge-text"),
  predictedPrice: document.getElementById("predicted-price"),
  predictedSubtext: document.getElementById("predicted-subtext"),
  retainedPercent: document.getElementById("retained-percent"),
  newPriceRef: document.getElementById("new-price-ref"),
  rangeLow: document.getElementById("range-low"),
  rangeHigh: document.getElementById("range-high"),
  summaryChips: document.getElementById("vehicle-summary-chips"),
  forecastChart: document.getElementById("forecast-chart"),
  impactsList: document.getElementById("impacts-list"),
  metricsGrid: document.getElementById("model-metrics-grid"),
  importanceHeading: document.getElementById("importance-heading"),
  globalImportanceList: document.getElementById("global-importance-list"),
  modelComparisonBody: document.getElementById("model-comparison-body"),
  historyList: document.getElementById("history-list"),
  historyCount: document.getElementById("history-count"),
  clearHistoryBtn: document.getElementById("clear-history-btn"),
};

function formatPrice(inrValue, compact = false) {
  if (state.currency === "USD") {
    const usd = Math.round(inrValue / INR_PER_USD);
    if (compact && usd >= 1000) {
      return `$${(usd / 1000).toFixed(1)}k`;
    }
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(usd);
  }

  if (compact) {
    if (inrValue >= 100000) {
      return `₹${(inrValue / 100000).toFixed(2)}L`;
    }
    return `₹${Math.round(inrValue / 1000)}k`;
  }

  return `₹ ${new Intl.NumberFormat("en-IN").format(Math.round(inrValue))}`;
}

function formatSubPrice(inrValue) {
  if (state.currency === "USD") {
    return `Approx. ₹ ${(inrValue / 100000).toFixed(2)} Lakh INR`;
  }
  return `₹${(inrValue / 100000).toFixed(2)} Lakh (${formatPrice(inrValue)})`;
}

function populateSelect(selectEl, options, preferredValue = null) {
  selectEl.innerHTML = "";
  options.forEach((opt) => {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt;
    selectEl.appendChild(el);
  });
  if (preferredValue && options.includes(preferredValue)) {
    selectEl.value = preferredValue;
  }
}

function updateModelDropdown(preferredModel = null) {
  const brand = dom.brandSelect.value;
  const modelsMap = state.catalog.brands[brand];
  const modelNames = Object.keys(modelsMap);
  populateSelect(dom.modelSelect, modelNames, preferredModel);
  syncModelSpecs();
}

function syncModelSpecs() {
  const brand = dom.brandSelect.value;
  const model = dom.modelSelect.value;
  const spec = state.catalog.brands[brand][model];

  dom.bodyBadge.textContent = spec.body_type;
  populateSelect(dom.fuelSelect, spec.fuels, dom.fuelSelect.value);
  populateSelect(dom.transmissionSelect, spec.transmissions, dom.transmissionSelect.value);

  const isElectric = dom.fuelSelect.value === "Electric";
  dom.engineCc.value = isElectric ? 0 : spec.engine_cc;
  dom.powerBhp.value = spec.max_power_bhp;
  dom.mileageKmpl.value = spec.mileage_kmpl;
}

function updateYearLabel() {
  const year = Number(dom.yearRange.value);
  const age = Math.max(0, state.catalog.current_year - year);
  dom.yearDisplay.textContent = String(year);
  dom.carAgeLabel.textContent = age === 1 ? "1 yr old" : `${age} yrs old`;
}

function renderConditionButtons(options) {
  dom.conditionGroup.innerHTML = "";
  options.forEach((cond) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `segment-btn${cond === state.condition ? " active" : ""}`;
    btn.textContent = cond;
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(cond === state.condition));
    btn.addEventListener("click", () => {
      state.condition = cond;
      renderConditionButtons(options);
      scheduleLivePredict();
    });
    dom.conditionGroup.appendChild(btn);
  });
}

function collectPayload() {
  return {
    brand: dom.brandSelect.value,
    model: dom.modelSelect.value,
    year: Number(dom.yearRange.value),
    km_driven: Number(dom.kmInput.value),
    fuel_type: dom.fuelSelect.value,
    transmission: dom.transmissionSelect.value,
    owner: dom.ownerSelect.value,
    seller_type: dom.sellerSelect.value,
    condition: state.condition,
    engine_cc: Number(dom.engineCc.value),
    max_power_bhp: Number(dom.powerBhp.value),
    mileage_kmpl: Number(dom.mileageKmpl.value),
  };
}

function showError(message) {
  if (!message) {
    dom.errorBanner.hidden = true;
    dom.errorBanner.textContent = "";
    return;
  }
  dom.errorBanner.hidden = false;
  dom.errorBanner.textContent = message;
}

function renderForecastSvg(forecast) {
  if (!forecast || forecast.length === 0) {
    dom.forecastChart.innerHTML = `<p class="empty-state">No forecast available.</p>`;
    return;
  }

  const width = 480;
  const height = 225;
  const pad = { top: 26, right: 26, bottom: 38, left: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const prices = forecast.map((d) => d.predicted_price);
  const maxP = Math.max(...prices) * 1.08;
  const minP = Math.max(0, Math.min(...prices) * 0.85);
  const span = Math.max(1, maxP - minP);

  const coords = forecast.map((d, i) => {
    const x = pad.left + (i / Math.max(1, forecast.length - 1)) * plotW;
    const y = pad.top + plotH - ((d.predicted_price - minP) / span) * plotH;
    return { x, y, ...d };
  });

  const linePoints = coords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const areaPath = `M ${coords[0].x.toFixed(1)},${(pad.top + plotH).toFixed(1)} L ${coords
    .map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(" L ")} L ${coords[coords.length - 1].x.toFixed(1)},${(pad.top + plotH).toFixed(1)} Z`;

  const gridLines = [0, 0.5, 1]
    .map((ratio) => {
      const y = pad.top + plotH - ratio * plotH;
      const val = minP + ratio * span;
      return `
        <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${width - pad.right}" y2="${y.toFixed(1)}" stroke="rgba(148,163,184,0.16)" stroke-dasharray="4 4" />
        <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="#64748b" font-size="11">${formatPrice(val, true)}</text>
      `;
    })
    .join("");

  const nodes = coords
    .map(
      (p, idx) => `
      <g>
        <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${idx === 0 ? 5.5 : 4}" fill="${idx === 0 ? "#38bdf8" : "#10b981"}" stroke="#090d16" stroke-width="2" />
        <text x="${p.x.toFixed(1)}" y="${(p.y - 11).toFixed(1)}" text-anchor="middle" fill="#e2e8f0" font-size="10.5" font-weight="600">${formatPrice(p.predicted_price, true)}</text>
        <text x="${p.x.toFixed(1)}" y="${height - 12}" text-anchor="middle" fill="#94a3b8" font-size="11">${p.year}</text>
      </g>
    `
    )
    .join("");

  dom.forecastChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="5-year resale value depreciation curve">
      <defs>
        <linearGradient id="depreciationArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#10b981" stop-opacity="0.34" />
          <stop offset="100%" stop-color="#10b981" stop-opacity="0.02" />
        </linearGradient>
      </defs>
      ${gridLines}
      <path d="${areaPath}" fill="url(#depreciationArea)" />
      <polyline points="${linePoints}" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
      ${nodes}
    </svg>
  `;
}

function renderImpacts(impacts) {
  dom.impactsList.innerHTML = "";
  if (!impacts || impacts.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Matches the baseline 5-year / 55,000 km specification closely.";
    dom.impactsList.appendChild(empty);
    return;
  }

  const maxAbs = Math.max(...impacts.map((item) => Math.abs(item.impact_inr)), 1);
  impacts.slice(0, 6).forEach((item) => {
    const positive = item.impact_inr >= 0;
    const widthPct = Math.max(8, Math.round((Math.abs(item.impact_inr) / maxAbs) * 100));

    const row = document.createElement("div");
    row.className = "impact-item";

    const meta = document.createElement("div");
    meta.className = "impact-meta";

    const label = document.createElement("span");
    label.className = "impact-label";
    label.textContent = item.label;

    const delta = document.createElement("span");
    delta.className = `impact-delta ${positive ? "positive" : "negative"}`;
    const sign = positive ? "+" : "-";
    delta.textContent = `${sign}${formatPrice(Math.abs(item.impact_inr), true)}`;

    meta.append(label, delta);

    const track = document.createElement("div");
    track.className = "impact-track";

    const bar = document.createElement("div");
    bar.className = `impact-bar ${positive ? "positive" : "negative"}`;
    bar.style.width = `${widthPct}%`;

    track.appendChild(bar);
    row.append(meta, track);
    dom.impactsList.appendChild(row);
  });
}

function renderPrediction(result) {
  state.lastResult = result;

  dom.predictedPrice.textContent = formatPrice(result.predicted_price);
  dom.predictedSubtext.textContent = formatSubPrice(result.predicted_price);
  dom.retainedPercent.textContent = `${result.retained_value_percent}%`;
  dom.newPriceRef.textContent = `of ${formatPrice(result.new_car_reference_price, true)} new`;
  dom.rangeLow.textContent = formatPrice(result.price_range.low);
  dom.rangeHigh.textContent = formatPrice(result.price_range.high);

  const v = result.vehicle;
  const chips = [
    `${v.brand} ${v.model}`,
    `${v.year}`,
    `${v.km_driven.toLocaleString("en-IN")} km`,
    v.fuel_type,
    v.transmission,
    v.owner,
    `${v.condition} Condition`,
  ];
  dom.summaryChips.innerHTML = "";
  chips.forEach((text) => {
    const chip = document.createElement("span");
    chip.className = "summary-chip";
    chip.textContent = text;
    dom.summaryChips.appendChild(chip);
  });

  renderForecastSvg(result.depreciation_forecast);
  renderImpacts(result.feature_impacts);
}

function renderModelTelemetry(metadata) {
  const m = metadata.metrics;
  dom.modelBadgeText.textContent = `${m.algorithm} • R² ${m.r2}`;
  dom.importanceHeading.textContent = `Global Feature Importance (${m.algorithm})`;

  const cards = [
    { label: "Test R² Score", value: m.r2 },
    { label: "Mean Abs. Error", value: formatPrice(m.mae, true) },
    { label: "Avg. Error (MAPE)", value: `${m.mape_percent}%` },
    { label: "Training Rows", value: m.train_samples.toLocaleString("en-IN") },
  ];

  dom.metricsGrid.innerHTML = "";
  cards.forEach((card) => {
    const box = document.createElement("div");
    box.className = "metric-box";
    const lbl = document.createElement("span");
    lbl.textContent = card.label;
    const val = document.createElement("strong");
    val.textContent = String(card.value);
    box.append(lbl, val);
    dom.metricsGrid.appendChild(box);
  });

  const topImportances = metadata.feature_importances.slice(0, 8);
  const maxImp = Math.max(...topImportances.map((x) => x.importance), 1);

  dom.globalImportanceList.innerHTML = "";
  topImportances.forEach((item) => {
    const row = document.createElement("div");
    row.className = "importance-row";

    const top = document.createElement("div");
    top.className = "importance-row-top";
    const name = document.createElement("span");
    name.textContent = item.label;
    const pct = document.createElement("strong");
    pct.textContent = `${item.importance}%`;
    top.append(name, pct);

    const track = document.createElement("div");
    track.className = "importance-track";
    const fill = document.createElement("div");
    fill.className = "importance-fill";
    fill.style.width = `${Math.max(5, Math.round((item.importance / maxImp) * 100))}%`;
    track.appendChild(fill);

    row.append(top, track);
    dom.globalImportanceList.appendChild(row);
  });

  if (dom.modelComparisonBody && Array.isArray(metadata.model_comparison)) {
    dom.modelComparisonBody.innerHTML = "";
    metadata.model_comparison.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.is_best) tr.className = "best-row";

      const cells = [
        row.algorithm,
        row.r2.toFixed(4),
        formatPrice(row.mae, true),
        formatPrice(row.rmse, true),
        `${row.mape_percent}%`,
      ];
      cells.forEach((val, idx) => {
        const td = document.createElement("td");
        td.textContent = val;
        if (idx === 0 && row.is_best) td.style.fontWeight = "700";
        tr.appendChild(td);
      });

      const statusTd = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = row.is_best ? "winner-pill" : "candidate-pill";
      badge.textContent = row.is_best ? "Selected" : "Benchmarked";
      statusTd.appendChild(badge);
      tr.appendChild(statusTd);

      dom.modelComparisonBody.appendChild(tr);
    });
  }
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveToHistory(result) {
  const v = result.vehicle;
  const entry = {
    vehicle: v,
    predicted_price: result.predicted_price,
    timestamp: Date.now(),
  };

  const existing = getHistory().filter(
    (item) =>
      !(
        item.vehicle.brand === v.brand &&
        item.vehicle.model === v.model &&
        item.vehicle.year === v.year &&
        item.vehicle.km_driven === v.km_driven &&
        item.vehicle.fuel_type === v.fuel_type &&
        item.vehicle.transmission === v.transmission
      )
  );

  const updated = [entry, ...existing].slice(0, MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  renderHistory();
}

function renderHistory() {
  const entries = getHistory();
  dom.historyCount.textContent = String(entries.length);
  dom.historyList.innerHTML = "";

  if (entries.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Click 'Estimate Resale Price' or a preset to save vehicles for comparison.";
    dom.historyList.appendChild(empty);
    return;
  }

  entries.forEach((entry) => {
    const v = entry.vehicle;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item";

    const info = document.createElement("div");
    const title = document.createElement("div");
    title.className = "history-title";
    title.textContent = `${v.year} ${v.brand} ${v.model}`;

    const sub = document.createElement("div");
    sub.className = "history-sub";
    sub.textContent = `${v.km_driven.toLocaleString("en-IN")} km • ${v.fuel_type} • ${v.transmission} • ${v.owner}`;
    info.append(title, sub);

    const price = document.createElement("div");
    price.className = "history-price";
    price.textContent = formatPrice(entry.predicted_price);

    btn.append(info, price);
    btn.addEventListener("click", () => applyVehicleConfig(v, false));
    dom.historyList.appendChild(btn);
  });
}

async function runPrediction(saveHistory = false) {
  showError("");
  dom.predictBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Failed to estimate price.");
      return;
    }
    renderPrediction(data);
    if (saveHistory) {
      saveToHistory(data);
    }
  } catch (err) {
    showError("Cannot reach the Python ML backend. Ensure the Flask server is running.");
  } finally {
    dom.predictBtn.disabled = false;
  }
}

function scheduleLivePredict() {
  if (!dom.liveToggle.checked) return;
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => runPrediction(false), 180);
}

function applyVehicleConfig(config, saveHistory = true) {
  dom.brandSelect.value = config.brand;
  updateModelDropdown(config.model);

  dom.yearRange.value = String(config.year);
  updateYearLabel();

  dom.kmRange.value = String(Math.min(220000, config.km_driven));
  dom.kmInput.value = String(config.km_driven);

  if (config.owner) dom.ownerSelect.value = config.owner;
  if (config.seller_type) dom.sellerSelect.value = config.seller_type;
  if (config.condition) {
    state.condition = config.condition;
    renderConditionButtons(state.catalog.conditions);
  }

  if (config.fuel_type) dom.fuelSelect.value = config.fuel_type;
  if (config.transmission) dom.transmissionSelect.value = config.transmission;

  if (config.engine_cc !== undefined) {
    dom.engineCc.value = String(config.engine_cc);
  } else if (dom.fuelSelect.value === "Electric") {
    dom.engineCc.value = "0";
  }
  if (config.max_power_bhp !== undefined) dom.powerBhp.value = String(config.max_power_bhp);
  if (config.mileage_kmpl !== undefined) dom.mileageKmpl.value = String(config.mileage_kmpl);

  runPrediction(saveHistory);
}

function bindEvents() {
  dom.brandSelect.addEventListener("change", () => {
    updateModelDropdown();
    scheduleLivePredict();
  });

  dom.modelSelect.addEventListener("change", () => {
    syncModelSpecs();
    scheduleLivePredict();
  });

  dom.fuelSelect.addEventListener("change", () => {
    const spec = state.catalog.brands[dom.brandSelect.value][dom.modelSelect.value];
    dom.engineCc.value = dom.fuelSelect.value === "Electric" ? "0" : String(spec.engine_cc);
    scheduleLivePredict();
  });

  [dom.transmissionSelect, dom.ownerSelect, dom.sellerSelect].forEach((el) => {
    el.addEventListener("change", scheduleLivePredict);
  });

  [dom.engineCc, dom.powerBhp, dom.mileageKmpl].forEach((el) => {
    el.addEventListener("input", scheduleLivePredict);
  });

  dom.yearRange.addEventListener("input", () => {
    updateYearLabel();
    scheduleLivePredict();
  });

  dom.kmRange.addEventListener("input", () => {
    dom.kmInput.value = dom.kmRange.value;
    scheduleLivePredict();
  });

  dom.kmInput.addEventListener("input", () => {
    const val = Math.max(0, Math.min(500000, Number(dom.kmInput.value) || 0));
    dom.kmRange.value = String(Math.min(220000, val));
    scheduleLivePredict();
  });

  dom.form.addEventListener("submit", (event) => {
    event.preventDefault();
    runPrediction(true);
  });

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = PRESETS[btn.dataset.preset];
      if (preset) applyVehicleConfig(preset, true);
    });
  });

  document.querySelectorAll(".currency-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.currency = btn.dataset.currency;
      document.querySelectorAll(".currency-btn").forEach((b) => b.classList.toggle("active", b === btn));
      if (state.lastResult) renderPrediction(state.lastResult);
      if (state.catalog) renderModelTelemetry(state.catalog.model_metadata);
      renderHistory();
    });
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => {
        const active = b === btn;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".tab-content").forEach((pane) => {
        pane.classList.toggle("active", pane.id === `tab-${target}`);
      });
    });
  });

  dom.clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });
}

async function init() {
  try {
    const response = await fetch(`${API_BASE}/api/catalog`);
    if (!response.ok) throw new Error("Failed to load catalog");
    state.catalog = await response.json();

    dom.yearRange.min = String(state.catalog.min_year);
    dom.yearRange.max = String(state.catalog.current_year);
    dom.minYearLabel.textContent = String(state.catalog.min_year);
    dom.maxYearLabel.textContent = String(state.catalog.current_year);

    populateSelect(dom.brandSelect, Object.keys(state.catalog.brands), "Hyundai");
    populateSelect(dom.ownerSelect, state.catalog.owners, "First Owner");
    populateSelect(dom.sellerSelect, state.catalog.sellers, "Dealer");
    renderConditionButtons(state.catalog.conditions);

    renderModelTelemetry(state.catalog.model_metadata);
    renderHistory();
    bindEvents();

    applyVehicleConfig(PRESETS.creta, true);
  } catch (err) {
    showError(
      "Cannot connect to the Python ML server at http://127.0.0.1:5000. Run 'python app.py' and open http://127.0.0.1:5000 in your browser."
    );
  }
}

init();
