import React, { useState, useEffect } from "react";
import type { OrderInput, PredictionResult, SampleScenario } from "./types";
import "./App.css";

const API_BASE = window.location.port === "5173" ? "http://127.0.0.1:8000" : "";

const DEFAULT_ORDER: OrderInput = {
  order_id: "KO2610504",
  customer_id: "KC100015",
  sku: "KH-RV-01",
  sales_channel: "marketplace",
  payment_mode: "cod",
  discount_pct: 15,
  qty: 1,
  promised_delivery_days: 8,
  delivery_pincode: 411001,
  is_gift: "N",
  customer_prior_orders: 3,
  customer_prior_returns: 2,
  delivery_note: "Ring twice, call before delivery",
};

const FALLBACK_SCENARIOS: SampleScenario[] = [
  {
    label: "🚨 High Risk (COD + Robot Vacuum + 2 Returns)",
    data: {
      order_id: "ORD-TEST-001",
      customer_id: "KC100015",
      sku: "KH-RV-01",
      sales_channel: "marketplace",
      payment_mode: "cod",
      discount_pct: 15,
      qty: 1,
      promised_delivery_days: 8,
      delivery_pincode: 411001,
      is_gift: "N",
      customer_prior_orders: 3,
      customer_prior_returns: 2,
      delivery_note: "Ring twice, call before delivery",
    },
  },
  {
    label: "⭐ Shield VIP Member (Air Fryer — Call, Don't Hold)",
    data: {
      order_id: "ORD-TEST-002",
      customer_id: "KC100021",
      sku: "KH-AF-02",
      sales_channel: "app",
      payment_mode: "prepaid_upi",
      discount_pct: 5,
      qty: 1,
      promised_delivery_days: 4,
      delivery_pincode: 400001,
      is_gift: "N",
      customer_prior_orders: 5,
      customer_prior_returns: 1,
      delivery_note: "Leave with security",
    },
  },
  {
    label: "✅ Low Risk (Prepaid UPI + Mixer Grinder)",
    data: {
      order_id: "ORD-TEST-003",
      customer_id: "KC100003",
      sku: "KH-MG-01",
      sales_channel: "web",
      payment_mode: "prepaid_upi",
      discount_pct: 10,
      qty: 1,
      promised_delivery_days: 2,
      delivery_pincode: 560001,
      is_gift: "N",
      customer_prior_orders: 2,
      customer_prior_returns: 0,
      delivery_note: "Office address, weekdays only",
    },
  },
  {
    label: "🏬 Walk-In Partner Outlet (Pincode 000000)",
    data: {
      order_id: "ORD-TEST-004",
      customer_id: "KC100100",
      sku: "KH-WP-01",
      sales_channel: "partner_outlet",
      payment_mode: "prepaid_card",
      discount_pct: 0,
      qty: 1,
      promised_delivery_days: 6,
      delivery_pincode: 0,
      is_gift: "N",
      customer_prior_orders: 0,
      customer_prior_returns: 0,
      delivery_note: "NONE",
    },
  },
];

export const App: React.FC = () => {
  const [formData, setFormData] = useState<OrderInput>(DEFAULT_ORDER);
  const [scenarios, setScenarios] = useState<SampleScenario[]>(FALLBACK_SCENARIOS);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/sample-orders`)
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) setScenarios(data);
      })
      .catch(() => {
        // Use fallback scenarios silently
      });

    // Run initial prediction
    handlePredict(DEFAULT_ORDER);
  }, []);

  const handlePredict = async (dataToPredict: OrderInput = formData) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dataToPredict),
      });
      if (!res.ok) {
        throw new Error(`API responded with status ${res.status}`);
      }
      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to connect to Kestrel Prediction Service.");
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? Number(value) : value,
    }));
  };

  const loadScenario = (scenario: SampleScenario) => {
    setFormData(scenario.data);
    handlePredict(scenario.data);
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case "VERY HIGH":
        return "#e53e3e";
      case "ELEVATED":
        return "#dd6b20";
      default:
        return "#38a169";
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-logo">🦅</span>
          <div>
            <h1>Kestrel Home — Pre-Dispatch Return Risk Desk</h1>
            <p className="subtitle">Real-Time Dispatch Risk Scoring & Operations Interventions</p>
          </div>
        </div>
        <div className="policy-badge">
          <span>Policy v4.1 Active</span>
          <small>Return Cost: Rs 1,150 | Call Cost: Rs 45</small>
        </div>
      </header>

      {/* Preset Scenarios */}
      <section className="scenarios-section">
        <span className="scenarios-label">⚡ Quick Test Scenarios:</span>
        <div className="scenarios-buttons">
          {scenarios.map((sc, idx) => (
            <button
              key={idx}
              className="scenario-btn"
              onClick={() => loadScenario(sc)}
              disabled={loading}
            >
              {sc.label}
            </button>
          ))}
        </div>
      </section>

      {/* Main Grid */}
      <div className="main-layout">
        {/* Left Column: Form */}
        <div className="card form-card">
          <h2>📦 Order Snapshot (At Dispatch)</h2>
          <p className="section-desc">Review and adjust order parameters before shipping.</p>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              handlePredict();
            }}
          >
            <div className="form-grid">
              <div className="field">
                <label>Order ID</label>
                <input
                  type="text"
                  name="order_id"
                  value={formData.order_id}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="field">
                <label>Customer ID</label>
                <input
                  type="text"
                  name="customer_id"
                  value={formData.customer_id}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="field">
                <label>SKU (Product)</label>
                <select name="sku" value={formData.sku} onChange={handleInputChange}>
                  <option value="KH-RV-01">KH-RV-01 (Robot Vacuum)</option>
                  <option value="KH-RV-02">KH-RV-02 (Robot Vacuum)</option>
                  <option value="KH-AF-01">KH-AF-01 (Air Fryer)</option>
                  <option value="KH-AF-02">KH-AF-02 (Air Fryer)</option>
                  <option value="KH-WP-01">KH-WP-01 (Water Purifier)</option>
                  <option value="KH-WP-02">KH-WP-02 (Water Purifier)</option>
                  <option value="KH-MG-01">KH-MG-01 (Mixer Grinder)</option>
                  <option value="KH-CF-01">KH-CF-01 (Ceiling Fan)</option>
                  <option value="KH-RH-01">KH-RH-01 (Room Heater)</option>
                  <option value="KH-IC-01">KH-IC-01 (Induction Cooktop)</option>
                </select>
              </div>

              <div className="field">
                <label>Payment Mode</label>
                <select name="payment_mode" value={formData.payment_mode} onChange={handleInputChange}>
                  <option value="cod">Cash on Delivery (COD)</option>
                  <option value="prepaid_upi">Prepaid UPI</option>
                  <option value="prepaid_card">Prepaid Card</option>
                  <option value="emi">EMI</option>
                </select>
              </div>

              <div className="field">
                <label>Sales Channel</label>
                <select name="sales_channel" value={formData.sales_channel} onChange={handleInputChange}>
                  <option value="marketplace">Marketplace</option>
                  <option value="app">Mobile App</option>
                  <option value="web">Website</option>
                  <option value="partner_outlet">Partner Outlet</option>
                </select>
              </div>

              <div className="field">
                <label>Promised Delivery Days</label>
                <input
                  type="number"
                  name="promised_delivery_days"
                  value={formData.promised_delivery_days}
                  onChange={handleInputChange}
                  min={1}
                  max={20}
                />
              </div>

              <div className="field">
                <label>Prior Customer Orders</label>
                <input
                  type="number"
                  name="customer_prior_orders"
                  value={formData.customer_prior_orders}
                  onChange={handleInputChange}
                  min={0}
                />
              </div>

              <div className="field">
                <label>Prior Customer Returns</label>
                <input
                  type="number"
                  name="customer_prior_returns"
                  value={formData.customer_prior_returns}
                  onChange={handleInputChange}
                  min={0}
                />
              </div>

              <div className="field">
                <label>Discount %</label>
                <input
                  type="number"
                  name="discount_pct"
                  value={formData.discount_pct}
                  onChange={handleInputChange}
                  min={0}
                  max={100}
                />
              </div>

              <div className="field">
                <label>Delivery Pincode</label>
                <input
                  type="number"
                  name="delivery_pincode"
                  value={formData.delivery_pincode}
                  onChange={handleInputChange}
                />
              </div>

              <div className="field">
                <label>Is Gift?</label>
                <select name="is_gift" value={formData.is_gift} onChange={handleInputChange}>
                  <option value="N">No (Personal)</option>
                  <option value="Y">Yes (Gift Order)</option>
                </select>
              </div>

              <div className="field full-width">
                <label>Delivery Note</label>
                <input
                  type="text"
                  name="delivery_note"
                  value={formData.delivery_note}
                  onChange={handleInputChange}
                />
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? "Evaluating Risk..." : "🔍 Inspect Order Return Risk"}
            </button>
          </form>
        </div>

        {/* Right Column: Prediction Results */}
        <div className="card result-card">
          <h2>📊 Risk Assessment & Action</h2>

          {error && <div className="error-banner">⚠️ {error}</div>}

          {result && (
            <div className="result-content">
              {/* Score Display */}
              <div className="score-hero">
                <div
                  className="score-gauge"
                  style={{ borderColor: getRiskColor(result.risk_level) }}
                >
                  <span className="score-number">{result.risk_percentage}</span>
                  <span className="score-label">RETURN PROBABILITY</span>
                </div>

                <div className="risk-badge-group">
                  <span
                    className="risk-badge"
                    style={{ backgroundColor: getRiskColor(result.risk_level) }}
                  >
                    {result.risk_level} RISK
                  </span>
                  {result.shield_member && (
                    <span className="shield-tag">🛡️ SHIELD VIP MEMBER</span>
                  )}
                </div>
              </div>

              {/* Action Banner */}
              <div className="action-box">
                <div className="action-title">RECOMMENDED WAREHOUSE ACTION:</div>
                <div className="action-desc">{result.recommended_action}</div>
              </div>

              {/* Shield Member Special Guidance */}
              {result.shield_member && (
                <div className="shield-guidance">
                  <strong>⚠️ VIP Retention Alert:</strong> This is a Shield member. Returns are free by policy. 
                  <em> DO NOT hold this dispatch.</em> If flagged, conduct a courteous confirmation call to confirm address and specs.
                </div>
              )}

              {/* Financial Box */}
              <div className="economics-box">
                <div className="econ-item">
                  <span className="econ-label">Calculated Order Value</span>
                  <span className="econ-val">Rs {result.order_value_inr.toLocaleString()}</span>
                </div>
                <div className="econ-item">
                  <span className="econ-label">Intervention Cost</span>
                  <span className="econ-val">
                    {result.return_risk_score >= 0.12 ? "Rs 45 (Call)" : "Rs 0"}
                  </span>
                </div>
                <div className="econ-item">
                  <span className="econ-label">Avoided Return Expense</span>
                  <span className="econ-val highlight">
                    {result.return_risk_score >= 0.12 ? "Rs 1,150" : "Baseline"}
                  </span>
                </div>
              </div>

              {/* Human-Readable Employee Reasons */}
              <div className="reasons-section">
                <h3>📝 Reasons For Warehouse Operator:</h3>
                <ul className="reasons-list">
                  {result.reasons.map((reason, idx) => (
                    <li key={idx} className="reason-item">
                      <span className="bullet">📌</span>
                      <span>{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
