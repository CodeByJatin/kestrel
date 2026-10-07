export interface OrderInput {
  order_id: string;
  customer_id: string;
  sku: string;
  sales_channel: string;
  payment_mode: string;
  discount_pct: number;
  qty: number;
  promised_delivery_days: number;
  delivery_pincode: number;
  is_gift: string;
  customer_prior_orders: number;
  customer_prior_returns: number;
  delivery_note: string;
  order_placed_at?: string;
}

export interface PredictionResult {
  order_id: string;
  return_risk_score: number;
  risk_percentage: string;
  risk_level: "LOW" | "ELEVATED" | "VERY HIGH";
  recommended_action: string;
  shield_member: boolean;
  order_value_inr: number;
  reasons: string[];
  metadata: {
    customer_id: string;
    sku: string;
    family: string;
    payment_mode: string;
    promised_delivery_days: number;
  };
}

export interface SampleScenario {
  label: string;
  data: OrderInput;
}
