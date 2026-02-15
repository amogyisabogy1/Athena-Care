import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Bell,
  Building2,
  ChevronDown,
  Download,
  Filter,
  LayoutDashboard,
  MapPin,
  Search,
  ShieldCheck,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// ui primitives
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Badge } from "./components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";
import { Tabs, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Separator } from "./components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";

/**
 * HealthScore AI – Main Dashboard
 * Single-file, production-ready scaffold with mock data.
 * - Left navigation
 * - KPI cards
 * - Risk forecast chart
 * - Alerts & AI insights
 * - Provider table preview
 */

const riskBadge = (score: number) => {
  if (score >= 80) return { label: "Low", variant: "secondary" as const };
  if (score >= 50) return { label: "Moderate", variant: "outline" as const };
  return { label: "High", variant: "destructive" as const };
};

const formatPct = (v: number) => `${(v * 100).toFixed(1)}%`;

const kpiDeltaBadge = (delta: number) => {
  const up = delta >= 0;
  return (
    <Badge variant={up ? "secondary" : "destructive"} className="ml-2">
      {up ? "+" : ""}
      {delta.toFixed(1)}%
    </Badge>
  );
};

// ---- XGBoost model meta (from your JSON export) ----
const MODEL_META = {
  name: "xgboost-denial-risk",
  best_iteration: 71,
  best_score_auc: 0.6945684122189337,
  feature_names: [
    "Provider Organization Name (Legal Business Name)_complete",
    "Employer Identification Number (EIN)_complete",
    "Provider First Line Business Practice Location Address_complete",
    "Provider Business Practice Location Address City Name_complete",
    "Provider Business Practice Location Address State Name_complete",
    "Provider Business Practice Location Address Postal Code_complete",
    "Provider Business Practice Location Address Telephone Number_complete",
    "Healthcare Provider Taxonomy Code_1_complete",
    "Provider License Number_1_complete",
    "Provider License Number State Code_1_complete",
    "data_completeness_score",
    "data_completeness_score.1",
    "missing_critical_fields",
    "num_taxonomy_codes",
    "hospital_type",
    "num_licenses",
    "has_primary_license",
    "license_state_match",
    "days_since_enumeration",
    "days_since_update",
    "recently_updated",
    "is_subpart",
    "has_parent_org",
    "state",
    "region",
  ],
} as const;

// Recommended production setup: host XGBoost behind an API and call it here.
// POST /api/predict should return:
//   { provider_key: string, denial_probability: number, top_factors?: {feature:string,impact:number}[] }
async function predictDenialRisk(payload: {
  provider_key: string;
  features: Record<string, number>;
  signal?: AbortSignal;
}) {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: payload.signal,
    body: JSON.stringify({
      model: MODEL_META.name,
      best_iteration: MODEL_META.best_iteration,
      provider_key: payload.provider_key,
      features: payload.features,
    }),
  });
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  return (await res.json()) as {
    provider_key: string;
    denial_probability: number;
    top_factors?: { feature: string; impact: number }[];
  };
}

function normalizeFeaturesForModel(features: Record<string, number>) {
  const out: Record<string, number> = {};
  for (const f of MODEL_META.feature_names) out[f] = features[f] ?? 0;
  return out;
}

function probToRiskScore(p: number) {
  const clamped = Math.min(0.999, Math.max(0.001, p));
  return Math.round((1 - clamped) * 100);
}

function riskLabelFromProb(p: number) {
  if (p >= 0.35) return { label: "High", variant: "destructive" as const };
  if (p >= 0.22) return { label: "Moderate", variant: "outline" as const };
  return { label: "Low", variant: "secondary" as const };
}

type Provider = {
  key: string;
  name: string;
  city: string;
  specialty: string;
  score: number;
  denialRate: number;
  predicted6m: number;
  features: Record<string, number>;
};

const FEATURE_LABELS: Record<string, string> = {
  missing_critical_fields: "Missing critical fields",
  data_completeness_score: "Data completeness score",
  "data_completeness_score.1": "Alt data completeness score",
  num_licenses: "Number of licenses",
  has_primary_license: "Primary license present",
  license_state_match: "License state match",
  days_since_update: "Days since update",
  recently_updated: "Recently updated",
  num_taxonomy_codes: "Taxonomy codes",
  hospital_type: "Hospital type",
};

const FEATURE_ACTIONS: Record<string, string> = {
  missing_critical_fields:
    "Resolve missing core fields (EIN, license, address) to reduce risk flags.",
  data_completeness_score:
    "Improve completeness score by validating NPI address and license records.",
  "data_completeness_score.1":
    "Align supplemental completeness feed with primary registry values.",
  num_licenses:
    "Verify license inventory and ensure active licensure across sites.",
  has_primary_license:
    "Confirm primary license is on file and active in current state.",
  license_state_match:
    "Check license state alignment with practice location.",
  days_since_update:
    "Refresh provider record to reduce staleness risk.",
  recently_updated:
    "Recent updates reduce denial risk; keep data fresh.",
  num_taxonomy_codes:
    "Validate taxonomy codes to ensure payer matching accuracy.",
  hospital_type:
    "Confirm hospital classification to avoid mismatched billing edits.",
};

const EDITABLE_FEATURES = [
  { key: "missing_critical_fields", min: 0, max: 5, step: 1 },
  { key: "data_completeness_score", min: 0, max: 1, step: 0.01 },
  { key: "num_licenses", min: 0, max: 6, step: 1 },
  { key: "has_primary_license", min: 0, max: 1, step: 1 },
  { key: "license_state_match", min: 0, max: 1, step: 1 },
  { key: "days_since_update", min: 0, max: 1000, step: 10 },
  { key: "recently_updated", min: 0, max: 1, step: 1 },
  { key: "num_taxonomy_codes", min: 0, max: 5, step: 1 },
  { key: "hospital_type", min: 0, max: 3, step: 1 },
];

const MOCK_FORECAST = [
  { m: "Mar", denial: 0.18, predicted: 0.19 },
  { m: "Apr", denial: 0.19, predicted: 0.205 },
  { m: "May", denial: 0.205, predicted: 0.22 },
  { m: "Jun", denial: 0.21, predicted: 0.235 },
  { m: "Jul", denial: 0.22, predicted: 0.26 },
  { m: "Aug", denial: 0.225, predicted: 0.275 },
  { m: "Sep", denial: 0.23, predicted: 0.285 },
  { m: "Oct", denial: 0.235, predicted: 0.29 },
];

const MOCK_PROVIDERS = [
  {
    key: "st-meridian",
    name: "St. Meridian Medical Center",
    city: "San Jose, CA",
    specialty: "Cardiology",
    score: 83,
    denialRate: 0.17,
    predicted6m: 0.19,
    features: {
      "Provider Organization Name (Legal Business Name)_complete": 1,
      "Employer Identification Number (EIN)_complete": 1,
      "Provider First Line Business Practice Location Address_complete": 1,
      "Provider Business Practice Location Address City Name_complete": 1,
      "Provider Business Practice Location Address State Name_complete": 1,
      "Provider Business Practice Location Address Postal Code_complete": 1,
      "Provider Business Practice Location Address Telephone Number_complete": 1,
      "Healthcare Provider Taxonomy Code_1_complete": 1,
      "Provider License Number_1_complete": 1,
      "Provider License Number State Code_1_complete": 1,
      data_completeness_score: 0.96,
      "data_completeness_score.1": 0.94,
      missing_critical_fields: 0,
      num_taxonomy_codes: 2,
      hospital_type: 2,
      num_licenses: 3,
      has_primary_license: 1,
      license_state_match: 1,
      days_since_enumeration: 3400,
      days_since_update: 18,
      recently_updated: 1,
      is_subpart: 0,
      has_parent_org: 1,
      state: 6,
      region: 4,
    } as Record<string, number>,
  },
  {
    key: "riverside-regional",
    name: "Riverside Regional Hospital",
    city: "Riverside, CA",
    specialty: "Orthopedics",
    score: 62,
    denialRate: 0.22,
    predicted6m: 0.27,
    features: {
      "Provider Organization Name (Legal Business Name)_complete": 1,
      "Employer Identification Number (EIN)_complete": 1,
      "Provider First Line Business Practice Location Address_complete": 1,
      "Provider Business Practice Location Address City Name_complete": 1,
      "Provider Business Practice Location Address State Name_complete": 1,
      "Provider Business Practice Location Address Postal Code_complete": 1,
      "Provider Business Practice Location Address Telephone Number_complete": 1,
      "Healthcare Provider Taxonomy Code_1_complete": 1,
      "Provider License Number_1_complete": 1,
      "Provider License Number State Code_1_complete": 1,
      data_completeness_score: 0.82,
      "data_completeness_score.1": 0.79,
      missing_critical_fields: 1,
      num_taxonomy_codes: 1,
      hospital_type: 2,
      num_licenses: 2,
      has_primary_license: 1,
      license_state_match: 1,
      days_since_enumeration: 5100,
      days_since_update: 220,
      recently_updated: 0,
      is_subpart: 0,
      has_parent_org: 1,
      state: 6,
      region: 4,
    } as Record<string, number>,
  },
  {
    key: "bayview-childrens",
    name: "Bayview Children’s",
    city: "Oakland, CA",
    specialty: "Pediatrics",
    score: 78,
    denialRate: 0.19,
    predicted6m: 0.21,
    features: {
      "Provider Organization Name (Legal Business Name)_complete": 1,
      "Employer Identification Number (EIN)_complete": 0,
      "Provider First Line Business Practice Location Address_complete": 1,
      "Provider Business Practice Location Address City Name_complete": 1,
      "Provider Business Practice Location Address State Name_complete": 1,
      "Provider Business Practice Location Address Postal Code_complete": 1,
      "Provider Business Practice Location Address Telephone Number_complete": 1,
      "Healthcare Provider Taxonomy Code_1_complete": 1,
      "Provider License Number_1_complete": 1,
      "Provider License Number State Code_1_complete": 1,
      data_completeness_score: 0.9,
      "data_completeness_score.1": 0.88,
      missing_critical_fields: 0,
      num_taxonomy_codes: 2,
      hospital_type: 1,
      num_licenses: 2,
      has_primary_license: 1,
      license_state_match: 1,
      days_since_enumeration: 2800,
      days_since_update: 44,
      recently_updated: 1,
      is_subpart: 0,
      has_parent_org: 0,
      state: 6,
      region: 4,
    } as Record<string, number>,
  },
  {
    key: "sierra-valley",
    name: "Sierra Valley Clinic Network",
    city: "Fresno, CA",
    specialty: "Primary Care",
    score: 49,
    denialRate: 0.28,
    predicted6m: 0.33,
    features: {
      "Provider Organization Name (Legal Business Name)_complete": 1,
      "Employer Identification Number (EIN)_complete": 0,
      "Provider First Line Business Practice Location Address_complete": 0,
      "Provider Business Practice Location Address City Name_complete": 1,
      "Provider Business Practice Location Address State Name_complete": 1,
      "Provider Business Practice Location Address Postal Code_complete": 0,
      "Provider Business Practice Location Address Telephone Number_complete": 0,
      "Healthcare Provider Taxonomy Code_1_complete": 1,
      "Provider License Number_1_complete": 0,
      "Provider License Number State Code_1_complete": 0,
      data_completeness_score: 0.55,
      "data_completeness_score.1": 0.52,
      missing_critical_fields: 3,
      num_taxonomy_codes: 1,
      hospital_type: 0,
      num_licenses: 0,
      has_primary_license: 0,
      license_state_match: 0,
      days_since_enumeration: 1900,
      days_since_update: 520,
      recently_updated: 0,
      is_subpart: 1,
      has_parent_org: 0,
      state: 6,
      region: 4,
    } as Record<string, number>,
  },
  {
    key: "northgate-oncology",
    name: "Northgate Oncology Institute",
    city: "Sacramento, CA",
    specialty: "Oncology",
    score: 71,
    denialRate: 0.21,
    predicted6m: 0.24,
    features: {
      "Provider Organization Name (Legal Business Name)_complete": 1,
      "Employer Identification Number (EIN)_complete": 1,
      "Provider First Line Business Practice Location Address_complete": 1,
      "Provider Business Practice Location Address City Name_complete": 1,
      "Provider Business Practice Location Address State Name_complete": 1,
      "Provider Business Practice Location Address Postal Code_complete": 1,
      "Provider Business Practice Location Address Telephone Number_complete": 1,
      "Healthcare Provider Taxonomy Code_1_complete": 1,
      "Provider License Number_1_complete": 1,
      "Provider License Number State Code_1_complete": 1,
      data_completeness_score: 0.86,
      "data_completeness_score.1": 0.84,
      missing_critical_fields: 0,
      num_taxonomy_codes: 2,
      hospital_type: 1,
      num_licenses: 1,
      has_primary_license: 1,
      license_state_match: 1,
      days_since_enumeration: 4100,
      days_since_update: 120,
      recently_updated: 0,
      is_subpart: 0,
      has_parent_org: 1,
      state: 6,
      region: 4,
    } as Record<string, number>,
  },
] as const;

const MOCK_ALERTS = [
  {
    title: "Risk spike predicted in Q3 for Riverside Regional",
    detail:
      "Drivers: payer policy shifts + elevated prior-auth friction in orthopedics.",
    severity: "High",
  },
  {
    title: "Denial rate trend drifting up across CA (last 60d)",
    detail: "Pattern consistent with tightening medical necessity edits.",
    severity: "Moderate",
  },
  {
    title: "Data freshness: NPI denial feed updated 2 hours ago",
    detail: "No gaps detected in ingestion pipeline.",
    severity: "Info",
  },
];

const MOCK_INSIGHTS = [
  {
    headline: "Partner shortlist suggestion",
    body:
      "For cardiology pilots, St. Meridian ranks top-tier (low risk, stable trend). Consider Bayview Children’s as a secondary site.",
    icon: <Sparkles className="h-4 w-4" />,
  },
  {
    headline: "Contracting guidance",
    body:
      "Add denial-rate SLA clauses for providers scoring < 60. Recommend quarterly review cadence for moderate-risk partners.",
    icon: <ShieldCheck className="h-4 w-4" />,
  },
  {
    headline: "Expansion signal",
    body:
      "Sacramento region shows improving denial trend (−1.2pp QoQ). Potentially favorable for new device rollout.",
    icon: <TrendingUp className="h-4 w-4" />,
  },
];

function SidebarItem({
  icon,
  label,
  active,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <button
      className={
        "w-full flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition " +
        (active
          ? "bg-white/10 text-white"
          : "text-white/70 hover:bg-white/5 hover:text-white")
      }
    >
      <span className="opacity-90">{icon}</span>
      <span className="truncate">{label}</span>
    </button>
  );
}

function KpiCard({
  title,
  value,
  sub,
  icon,
}: {
  title: string;
  value: React.ReactNode;
  sub: React.ReactNode;
  icon: React.ReactNode;
}) {
  return (
    <Card className="rounded-2xl border-white/10 bg-white/5">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-white/70 flex items-center justify-between">
          <span>{title}</span>
          <span className="text-white/60">{icon}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold tracking-tight text-white">
          {value}
        </div>
        <div className="mt-1 text-xs text-white/60">{sub}</div>
      </CardContent>
    </Card>
  );
}

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const p = payload.reduce((acc: any, cur: any) => {
    acc[cur.dataKey] = cur.value;
    return acc;
  }, {});
  return (
    <div className="rounded-xl border border-white/10 bg-zinc-950/80 px-3 py-2 text-xs text-white shadow-lg backdrop-blur">
      <div className="font-medium mb-1">{label}</div>
      {p.denial != null && (
        <div className="flex items-center justify-between gap-4">
          <span className="text-white/70">Observed denial</span>
          <span className="font-medium">{formatPct(p.denial)}</span>
        </div>
      )}
      {p.predicted != null && (
        <div className="flex items-center justify-between gap-4">
          <span className="text-white/70">Predicted denial</span>
          <span className="font-medium">{formatPct(p.predicted)}</span>
        </div>
      )}
    </div>
  );
}

export default function HealthScoreAIDashboard() {
  const [modelEnabled, setModelEnabled] = useState(true);
  const [selectedKey, setSelectedKey] = useState(MOCK_PROVIDERS[0].key);
  const [scenarioFeatures, setScenarioFeatures] = useState<Record<string, number>>(
    () => normalizeFeaturesForModel(MOCK_PROVIDERS[0].features)
  );
  const [scenarioResult, setScenarioResult] = useState<
    | {
        provider_key: string;
        denial_probability: number;
        top_factors?: { feature: string; impact: number }[];
      }
    | null
  >(null);
  const [scenarioError, setScenarioError] = useState<string | null>(null);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [scenarioUpdatedAt, setScenarioUpdatedAt] = useState<string | null>(null);
  const [showControls, setShowControls] = useState(true);

  const [tab, setTab] = useState<"overview" | "network">("overview");
  const [query, setQuery] = useState("");
  const [region, setRegion] = useState("United States");

  const providers = useMemo(() => MOCK_PROVIDERS as Provider[], []);
  const selectedProvider = useMemo(
    () => providers.find((p) => p.key === selectedKey) ?? providers[0],
    [providers, selectedKey]
  );

  const filteredProviders = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return MOCK_PROVIDERS;
    return MOCK_PROVIDERS.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.city.toLowerCase().includes(q) ||
        p.specialty.toLowerCase().includes(q)
    );
  }, [query]);

  useEffect(() => {
    setScenarioFeatures(normalizeFeaturesForModel(selectedProvider.features));
  }, [selectedProvider]);

  useEffect(() => {
    if (!modelEnabled) return;
    if (!selectedProvider) return;

    const controller = new AbortController();
    const timeout = window.setTimeout(async () => {
      setScenarioLoading(true);
      setScenarioError(null);
      try {
        const result = await predictDenialRisk({
          provider_key: selectedProvider.key,
          features: normalizeFeaturesForModel(scenarioFeatures),
          signal: controller.signal,
        });
        setScenarioResult(result);
        setScenarioUpdatedAt(new Date().toLocaleTimeString());
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setScenarioError(
          err instanceof Error
            ? err.message
            : "Prediction failed. Check backend connectivity."
        );
      } finally {
        setScenarioLoading(false);
      }
    }, 350);

    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [modelEnabled, scenarioFeatures, selectedProvider]);

  // Example KPIs (mock)
  const avgScore = useMemo(() => {
    if (modelEnabled && scenarioResult) {
      return probToRiskScore(scenarioResult.denial_probability);
    }

    const arr = providers.map((p) => p.score);
    return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
  }, [modelEnabled, scenarioResult, providers]);

  const highRiskCount = useMemo(() => {
    if (modelEnabled && scenarioResult) {
      return scenarioResult.denial_probability >= 0.35 ? 1 : 0;
    }
    return providers.filter((p) => p.score < 50).length;
  }, [modelEnabled, scenarioResult, providers]);

  const forecastData = useMemo(() => {
    const base = scenarioResult?.denial_probability ?? MOCK_FORECAST[3].denial;
    return MOCK_FORECAST.map((d, idx) => ({
      ...d,
      denial: Math.max(0.05, base - 0.02 + idx * 0.003),
      predicted: Math.min(0.6, base + 0.01 + idx * 0.008),
    }));
  }, [scenarioResult]);

  const clinicianBrief = useMemo(() => {
    if (!scenarioResult?.top_factors?.length) return [] as string[];
    return scenarioResult.top_factors
      .slice(0, 3)
      .map((f) => FEATURE_ACTIONS[f.feature])
      .filter(Boolean);
  }, [scenarioResult]);

  const predictedSpike = 14.0; // percent points, mock

  const onResetScenario = () => {
    setScenarioFeatures(normalizeFeaturesForModel(selectedProvider.features));
  };

  const onExportScenario = () => {
    const payload = {
      provider: selectedProvider,
      features: scenarioFeatures,
      prediction: scenarioResult,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${selectedProvider.key}-scenario.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const clinicianSummaryText = useMemo(() => {
    if (!scenarioResult) return "No prediction available yet.";
    const risk = riskLabelFromProb(scenarioResult.denial_probability).label;
    const factors = scenarioResult.top_factors?.
      slice(0, 3)
      .map((f) => FEATURE_LABELS[f.feature] ?? f.feature)
      .join(", ");
    return `Provider ${selectedProvider.name} has ${risk} denial risk ` +
      `(${formatPct(scenarioResult.denial_probability)}). ` +
      (factors ? `Top drivers: ${factors}.` : "");
  }, [scenarioResult, selectedProvider]);

  const onCopyBrief = async () => {
    try {
      await navigator.clipboard.writeText(clinicianSummaryText);
    } catch {
      // ignore clipboard errors silently
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      <div className="flex">
        {/* Sidebar */}
        <aside className="hidden md:flex w-[280px] flex-col border-r border-white/10 bg-black/40 px-4 py-5">
          <div className="flex items-center gap-3 px-2">
            <div className="h-10 w-10 rounded-2xl bg-white/10 flex items-center justify-center">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="text-sm font-semibold leading-tight">
                HealthScore AI
              </div>
              <div className="text-xs text-white/60">Provider Risk Intel</div>
            </div>
          </div>

          <Separator className="my-4 bg-white/10" />

          <nav className="space-y-1">
            <SidebarItem
              icon={<LayoutDashboard className="h-4 w-4" />}
              label="Dashboard"
              active
            />
            <SidebarItem
              icon={<Search className="h-4 w-4" />}
              label="Provider Explorer"
            />
            <SidebarItem
              icon={<TrendingUp className="h-4 w-4" />}
              label="Predictions"
            />
            <SidebarItem
              icon={<Bell className="h-4 w-4" />}
              label="Alerts"
            />
            <SidebarItem
              icon={<Download className="h-4 w-4" />}
              label="Reports"
            />
            <SidebarItem
              icon={<ShieldCheck className="h-4 w-4" />}
              label="API / Data"
            />
          </nav>

          <div className="mt-auto pt-4">
            <Card className="rounded-2xl border-white/10 bg-white/5">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-white/70">Trust & Compliance</div>
                  <Badge variant="secondary">SOC 2</Badge>
                </div>
                <div className="mt-2 text-xs text-white/60">
                  Data freshness: <span className="text-white/80">2 hours</span>
                </div>
                <div className="mt-1 text-xs text-white/60">
                  Coverage: <span className="text-white/80">Hospitals + Clinics</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1">
          {/* Top bar */}
          <div className="sticky top-0 z-20 border-b border-white/10 bg-zinc-950/70 backdrop-blur">
            <div className="mx-auto max-w-7xl px-4 py-4">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="text-lg font-semibold">Dashboard</div>
                  <div className="text-xs text-white/60">
                    Know before you partner • Predict denials 6–12 months ahead
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative w-full md:w-[340px]">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/50" />
                    <Input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search provider, city, specialty"
                      className="h-10 rounded-2xl border-white/10 bg-white/5 pl-9 text-white placeholder:text-white/40"
                    />
                  </div>

                  <DropdownMenu>
                    <DropdownMenuTrigger>
                      <Button
                        variant="secondary"
                        className="h-10 rounded-2xl bg-white/10 hover:bg-white/15"
                      >
                        <MapPin className="h-4 w-4 mr-2" />
                        {region}
                        <ChevronDown className="h-4 w-4 ml-2 opacity-70" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="border-white/10 bg-zinc-950 text-white">
                      <DropdownMenuLabel className="text-white/70">
                        Region
                      </DropdownMenuLabel>
                      <DropdownMenuSeparator className="bg-white/10" />
                      {[
                        "United States",
                        "California",
                        "Northeast",
                        "Midwest",
                        "South",
                      ].map((r) => (
                        <DropdownMenuItem
                          key={r}
                          onClick={() => setRegion(r)}
                          className="focus:bg-white/10 focus:text-white"
                        >
                          {r}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>

                  <Button
                    variant="outline"
                    className="h-10 rounded-2xl border-white/10 bg-transparent text-white hover:bg-white/5"
                  >
                    <Filter className="h-4 w-4 mr-2" />
                    Filters
                  </Button>

                  <Button
                    variant={modelEnabled ? "secondary" : "outline"}
                    className={
                      "h-10 rounded-2xl " +
                      (modelEnabled
                        ? "bg-white/10 hover:bg-white/15"
                        : "border-white/10 bg-transparent text-white hover:bg-white/5")
                    }
                    onClick={() => setModelEnabled((v) => !v)}
                    title="Toggle XGBoost model-driven scoring"
                  >
                    <Sparkles className="h-4 w-4 mr-2" />
                    Model {modelEnabled ? "On" : "Off"}
                  </Button>

                  <Button className="h-10 rounded-2xl">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between">
                <Tabs value={tab} onValueChange={(v) => setTab(v as any)}>
                  <TabsList className="rounded-2xl bg-white/5">
                    <TabsTrigger value="overview" className="rounded-xl">
                      Overview
                    </TabsTrigger>
                    <TabsTrigger value="network" className="rounded-xl">
                      Network
                    </TabsTrigger>
                  </TabsList>
                </Tabs>

                <div className="hidden md:flex items-center gap-2 text-xs text-white/60">
                  <Building2 className="h-4 w-4" />
                  Coverage: <span className="text-white/80">{filteredProviders.length}</span>
                  providers shown
                </div>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="mx-auto max-w-7xl px-4 py-6">
            {/* KPIs */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
              className="grid grid-cols-1 gap-4 md:grid-cols-4"
            >
              <KpiCard
                title="Avg Partner Risk Score"
                value={
                  <div className="flex items-center gap-2">
                    <span>{avgScore}/100</span>
                    <Badge variant="secondary" className="bg-white/10">
                      Medium
                    </Badge>
                  </div>
                }
                sub={
                  <span>
                    Based on recent NPI denial signals • Last 30 days
                    {kpiDeltaBadge(1.8)}
                  </span>
                }
                icon={<ShieldCheck className="h-4 w-4" />}
              />
              <KpiCard
                title="High-Risk Providers Flagged"
                value={
                  <div className="flex items-center gap-2">
                    <span>{highRiskCount}</span>
                    <Badge variant="destructive">Action</Badge>
                  </div>
                }
                sub={<span>Score &lt; 50 • Review contracting & workflows</span>}
                icon={<Bell className="h-4 w-4" />}
              />
              <KpiCard
                title="Predicted Denial Spike"
                value={
                  <div className="flex items-center gap-2">
                    <span>+{predictedSpike.toFixed(0)}%</span>
                    <Badge variant="outline" className="border-white/10">
                      Next 6 months
                    </Badge>
                  </div>
                }
                sub={<span>Forecasted trend from model ensemble</span>}
                icon={<TrendingUp className="h-4 w-4" />}
              />
              <KpiCard
                title="Data Pipeline Status"
                value={
                  <div className="flex items-center gap-2">
                    <span>Healthy</span>
                    <Badge variant="secondary" className="bg-white/10">
                      Live
                    </Badge>
                  </div>
                }
                sub={<span>Freshness 2h • Drift checks passing</span>}
                icon={<Sparkles className="h-4 w-4" />}
              />
            </motion.div>

            {/* Main grid */}
            <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
              {/* Forecast */}
              <Card className="rounded-2xl border-white/10 bg-white/5 lg:col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80 flex items-center justify-between">
                    <span>Denial Rate Forecast (6–12 months)</span>
                    <Badge variant="outline" className="border-white/10 text-white/80">
                      XGBoost • best_iter {MODEL_META.best_iteration} • AUC {MODEL_META.best_score_auc.toFixed(3)}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="h-[260px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={forecastData} margin={{ left: 6, right: 6, top: 10 }}>
                        <defs>
                          <linearGradient id="predFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="currentColor" stopOpacity={0.35} />
                            <stop offset="100%" stopColor="currentColor" stopOpacity={0.05} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis
                          dataKey="m"
                          tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
                          axisLine={{ stroke: "rgba(255,255,255,0.12)" }}
                          tickLine={{ stroke: "rgba(255,255,255,0.12)" }}
                        />
                        <YAxis
                          tickFormatter={(v) => `${Math.round(v * 100)}%`}
                          tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
                          axisLine={{ stroke: "rgba(255,255,255,0.12)" }}
                          tickLine={{ stroke: "rgba(255,255,255,0.12)" }}
                        />
                        <Tooltip content={<ChartTooltip />} />
                        <Area
                          type="monotone"
                          dataKey="predicted"
                          stroke="rgba(255,255,255,0.85)"
                          fill="url(#predFill)"
                          strokeWidth={2}
                        />
                        <Line
                          type="monotone"
                          dataKey="denial"
                          stroke="rgba(255,255,255,0.35)"
                          strokeWidth={2}
                          dot={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-xs text-white/60">Current denial</div>
                      <div className="mt-1 text-base font-semibold">
                        {formatPct(forecastData[3].denial)}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-xs text-white/60">Projected peak</div>
                      <div className="mt-1 text-base font-semibold">
                        {formatPct(Math.max(...forecastData.map((d) => d.predicted)))}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-xs text-white/60">Confidence</div>
                      <div className="mt-1 text-base font-semibold">High</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Alerts */}
              <Card className="rounded-2xl border-white/10 bg-white/5">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80 flex items-center justify-between">
                    <span>Active Alerts</span>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-8 rounded-xl border-white/10 bg-transparent text-white hover:bg-white/5"
                    >
                      View all
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {scenarioError && (
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3 text-xs text-white/70">
                      <div className="font-medium mb-1">Model status</div>
                      <div className="text-white/60">{scenarioError}</div>
                    </div>
                  )}
                  {!scenarioError && (
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3 text-xs text-white/70">
                      <div className="font-medium mb-1">Model status</div>
                      <div className="text-white/60">
                        {modelEnabled
                          ? scenarioLoading
                            ? "Updating scenario prediction…"
                            : scenarioUpdatedAt
                              ? `Updated at ${scenarioUpdatedAt}`
                              : "Ready"
                          : "Model disabled"}
                      </div>
                    </div>
                  )}
                  {MOCK_ALERTS.map((a, idx) => {
                    const sev = a.severity;
                    const badgeVariant =
                      sev === "High"
                        ? ("destructive" as const)
                        : sev === "Moderate"
                          ? ("outline" as const)
                          : ("secondary" as const);
                    return (
                      <div
                        key={idx}
                        className="rounded-2xl border border-white/10 bg-black/20 p-3"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium leading-snug">
                              {a.title}
                            </div>
                            <div className="mt-1 text-xs text-white/60">{a.detail}</div>
                          </div>
                          <Badge variant={badgeVariant} className="shrink-0">
                            {sev}
                          </Badge>
                        </div>
                      </div>
                    );
                  })}
                </CardContent>
              </Card>
            </div>

            {/* Bottom grid */}
            <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
              {/* Scenario Editor & Clinician Brief */}
              <Card className="rounded-2xl border-white/10 bg-white/5">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80 flex items-center justify-between">
                    <span>Scenario Editor</span>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 rounded-lg border-white/10 bg-transparent text-white hover:bg-white/5 text-xs px-2"
                      onClick={() => setShowControls(!showControls)}
                    >
                      {showControls ? "Hide" : "Show"}
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="rounded-lg bg-black/20 p-3">
                    <div className="text-xs text-white/60 mb-2">Select Provider</div>
                    <select
                      value={selectedKey}
                      onChange={(e) => setSelectedKey(e.target.value)}
                      className="w-full rounded-lg bg-white/10 border border-white/10 text-white text-sm p-2"
                    >
                      {providers.map((p) => (
                        <option key={p.key} value={p.key}>
                          {p.name} ({p.specialty})
                        </option>
                      ))}
                    </select>
                  </div>

                  {showControls && (
                    <div className="space-y-2 max-h-[360px] overflow-y-auto">
                      {EDITABLE_FEATURES.map(({ key: fkey, min, max, step }) => (
                        <div key={fkey} className="text-xs">
                          <div className="text-white/70 font-medium mb-1">
                            {FEATURE_LABELS[fkey] ?? fkey}
                          </div>
                          <div className="flex gap-2 items-center">
                            <input
                              type="range"
                              min={min}
                              max={max}
                              step={step}
                              value={scenarioFeatures[fkey] ?? 0}
                              onChange={(e) =>
                                setScenarioFeatures({
                                  ...scenarioFeatures,
                                  [fkey]: parseFloat(e.target.value),
                                })
                              }
                              className="flex-1"
                            />
                            <span className="text-white/60 w-12 text-right">
                              {(scenarioFeatures[fkey] ?? 0).toFixed(2)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="flex gap-2 pt-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 rounded-lg border-white/10 h-8"
                      onClick={onResetScenario}
                    >
                      Reset
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 rounded-lg border-white/10 h-8"
                      onClick={onExportScenario}
                    >
                      Export
                    </Button>
                  </div>

                  <div className="rounded-lg bg-black/30 p-3 border border-white/10">
                    <div className="text-xs text-white/60 mb-2">Clinician Brief</div>
                    <p className="text-xs text-white leading-relaxed mb-2">
                      {clinicianSummaryText}
                    </p>
                    <div className="flex gap-2">
                      {clinicianBrief.length > 0 && (
                        <div className="flex-1 text-[11px] text-white/70 leading-snug">
                          <div className="font-medium mb-1">Recommended actions:</div>
                          <ul className="list-disc list-inside space-y-1">
                            {clinicianBrief.map((action, idx) => (
                              <li key={idx}>{action}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full rounded-lg border-white/10 h-7 text-xs mt-2"
                      onClick={onCopyBrief}
                    >
                      Copy Brief
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* AI Insights */}
              <Card className="rounded-2xl border-white/10 bg-white/5 lg:col-span-1">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80">
                    AI Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {scenarioResult?.top_factors?.slice(0, 3).map((factor, idx) => {
                    const label = FEATURE_LABELS[factor.feature] ?? factor.feature;
                    const action = FEATURE_ACTIONS[factor.feature] ?? "Monitor this factor.";
                    return (
                      <div
                        key={idx}
                        className="rounded-2xl border border-white/10 bg-black/20 p-3"
                      >
                        <div className="flex items-center gap-2 text-sm font-medium">
                          <span className="text-blue-400">→</span>
                          <span>{label}</span>
                        </div>
                        <div className="mt-1 text-xs text-white/60 leading-relaxed">
                          {action}
                        </div>
                        <div className="mt-2 text-xs text-white/40">
                          Impact score: {factor.impact.toFixed(4)}
                        </div>
                      </div>
                    );
                  }) || (
                    <div className="text-xs text-white/50 text-center py-4">
                      Run scenario to see factor insights
                    </div>
                  )}
                  <Button className="w-full rounded-2xl" disabled={!scenarioResult}>
                    <Download className="h-4 w-4 mr-2" />
                    Download risk report
                  </Button>
                </CardContent>
              </Card>

              {/* Provider preview table */}
              <Card className="rounded-2xl border-white/10 bg-white/5 lg:col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80 flex items-center justify-between">
                    <span>{selectedProvider.name} – Scenario Prediction</span>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-8 rounded-xl border-white/10 bg-transparent text-white hover:bg-white/5"
                    >
                      Compare
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {scenarioLoading && (
                    <div className="text-center py-8 text-white/60">
                      Calculating prediction...
                    </div>
                  )}
                  {!scenarioLoading && scenarioResult && (
                    <div className="space-y-4">
                      <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div>
                            <div className="text-xs text-white/60 mb-1">Predicted denial probability</div>
                            <div className="text-2xl font-bold text-white">
                              {formatPct(scenarioResult.denial_probability)}
                            </div>
                            <Badge
                              variant={
                                riskLabelFromProb(scenarioResult.denial_probability).variant
                              }
                              className="mt-2"
                            >
                              {riskLabelFromProb(scenarioResult.denial_probability).label}
                            </Badge>
                          </div>
                          <div>
                            <div className="text-xs text-white/60 mb-1">Risk Score (0-100)</div>
                            <div className="text-2xl font-bold text-white">
                              {probToRiskScore(scenarioResult.denial_probability)}/100
                            </div>
                            <div className="text-xs text-white/50 mt-2">
                              Based on 25 clinical factors
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-white/60 mb-1">Most impactful factors</div>
                            <div className="space-y-1 text-xs text-white/70">
                              {scenarioResult.top_factors?.slice(0, 2).map((f, idx) => (
                                <div key={idx}>
                                  {(idx + 1)}. {FEATURE_LABELS[f.feature] ?? f.feature}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                        <div className="text-sm font-semibold text-white mb-3">
                          Scenario configuration
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                          {Object.entries(scenarioFeatures)
                            .slice(0, 6)
                            .map(([key, value]) => (
                              <div key={key}>
                                <div className="text-white/60 mb-1">
                                  {FEATURE_LABELS[key] ?? key}
                                </div>
                                <div className="text-white font-medium">
                                  {typeof value === "number" && value < 1
                                    ? value.toFixed(2)
                                    : Math.round(value)}
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                    </div>
                  )}
                  {!scenarioLoading && !scenarioResult && (
                    <div className="text-center py-8 text-white/60">
                      Adjust scenario parameters to generate a prediction
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            <div className="mt-6 text-xs text-white/50">
              Tip: this dashboard is wired with mock data. Replace MOCK_* with your API responses and feed them into the charts/tables.
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
