import React, { useMemo, useState } from "react";
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

// shadcn/ui
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

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
    name: "St. Meridian Medical Center",
    city: "San Jose, CA",
    specialty: "Cardiology",
    score: 83,
    denialRate: 0.17,
    predicted6m: 0.19,
  },
  {
    name: "Riverside Regional Hospital",
    city: "Riverside, CA",
    specialty: "Orthopedics",
    score: 62,
    denialRate: 0.22,
    predicted6m: 0.27,
  },
  {
    name: "Bayview Children’s",
    city: "Oakland, CA",
    specialty: "Pediatrics",
    score: 78,
    denialRate: 0.19,
    predicted6m: 0.21,
  },
  {
    name: "Sierra Valley Clinic Network",
    city: "Fresno, CA",
    specialty: "Primary Care",
    score: 49,
    denialRate: 0.28,
    predicted6m: 0.33,
  },
  {
    name: "Northgate Oncology Institute",
    city: "Sacramento, CA",
    specialty: "Oncology",
    score: 71,
    denialRate: 0.21,
    predicted6m: 0.24,
  },
];

const MOCK_ALERTS = [
  {
    title: "Risk spike predicted in Q3 for Riverside Regional",
    detail:
      "Drivers: payer policy shifts + elevated prior-auth friction in orthopedics.",
    severity: "High",
  },
  {
    title: "Denial rate trend drifting up across CA (last 60d)",
    detail:
      "Pattern consistent with tightening medical necessity edits.",
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
  const [tab, setTab] = useState<"overview" | "network">("overview");
  const [query, setQuery] = useState("");
  const [region, setRegion] = useState("United States");

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

  // Example KPIs (mock)
  const avgScore = useMemo(() => {
    const arr = MOCK_PROVIDERS.map((p) => p.score);
    return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
  }, []);

  const highRiskCount = useMemo(
    () => MOCK_PROVIDERS.filter((p) => p.score < 50).length,
    []
  );

  const predictedSpike = 14.0; // percent points, mock

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
                    <DropdownMenuTrigger asChild>
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
                      Model: v0.9
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="h-[260px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={MOCK_FORECAST} margin={{ left: 6, right: 6, top: 10 }}>
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
                        {formatPct(MOCK_FORECAST[3].denial)}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-xs text-white/60">Projected peak</div>
                      <div className="mt-1 text-base font-semibold">
                        {formatPct(Math.max(...MOCK_FORECAST.map((d) => d.predicted)))}
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
              {/* AI Insights */}
              <Card className="rounded-2xl border-white/10 bg-white/5 lg:col-span-1">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80">
                    AI Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {MOCK_INSIGHTS.map((ins, idx) => (
                    <div
                      key={idx}
                      className="rounded-2xl border border-white/10 bg-black/20 p-3"
                    >
                      <div className="flex items-center gap-2 text-sm font-medium">
                        <span className="text-white/70">{ins.icon}</span>
                        <span>{ins.headline}</span>
                      </div>
                      <div className="mt-1 text-xs text-white/60 leading-relaxed">
                        {ins.body}
                      </div>
                    </div>
                  ))}
                  <Button className="w-full rounded-2xl">
                    <Sparkles className="h-4 w-4 mr-2" />
                    Generate partner memo
                  </Button>
                </CardContent>
              </Card>

              {/* Provider preview table */}
              <Card className="rounded-2xl border-white/10 bg-white/5 lg:col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80 flex items-center justify-between">
                    <span>Provider Watchlist</span>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-8 rounded-xl border-white/10 bg-transparent text-white hover:bg-white/5"
                    >
                      Open explorer
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="rounded-2xl border border-white/10 overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-white/10 hover:bg-transparent">
                          <TableHead className="text-white/60">Provider</TableHead>
                          <TableHead className="text-white/60">Specialty</TableHead>
                          <TableHead className="text-white/60">Risk</TableHead>
                          <TableHead className="text-white/60 text-right">
                            Denial
                          </TableHead>
                          <TableHead className="text-white/60 text-right">
                            Pred (6m)
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredProviders.map((p) => {
                          const rb = riskBadge(p.score);
                          return (
                            <TableRow
                              key={p.name}
                              className="border-white/10 hover:bg-white/5"
                            >
                              <TableCell>
                                <div className="font-medium leading-snug">{p.name}</div>
                                <div className="text-xs text-white/60">{p.city}</div>
                              </TableCell>
                              <TableCell className="text-white/80">
                                {p.specialty}
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  <Badge variant={rb.variant}>{rb.label}</Badge>
                                  <span className="text-sm text-white/80">
                                    {p.score}/100
                                  </span>
                                </div>
                              </TableCell>
                              <TableCell className="text-right text-white/80">
                                {formatPct(p.denialRate)}
                              </TableCell>
                              <TableCell className="text-right text-white/80">
                                {formatPct(p.predicted6m)}
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>

                  <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
                    <Card className="rounded-2xl border-white/10 bg-black/20">
                      <CardContent className="p-4">
                        <div className="text-xs text-white/60">
                          Coverage snapshot
                        </div>
                        <div className="mt-2 h-[86px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={[
                                { x: "W1", v: 78 },
                                { x: "W2", v: 76 },
                                { x: "W3", v: 74 },
                                { x: "W4", v: 72 },
                                { x: "W5", v: 73 },
                              ]}
                            >
                              <XAxis hide dataKey="x" />
                              <YAxis hide />
                              <Tooltip contentStyle={{ display: "none" }} />
                              <Line
                                type="monotone"
                                dataKey="v"
                                stroke="rgba(255,255,255,0.7)"
                                strokeWidth={2}
                                dot={false}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="mt-2 text-sm font-semibold">
                          Network risk drifting
                        </div>
                        <div className="text-xs text-white/60">
                          Watch for Q3 payer tightening
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="rounded-2xl border-white/10 bg-black/20">
                      <CardContent className="p-4">
                        <div className="text-xs text-white/60">Hotspots</div>
                        <div className="mt-2 text-sm font-semibold">
                          Orthopedics • Prior auth
                        </div>
                        <div className="mt-1 text-xs text-white/60 leading-relaxed">
                          Elevations in denial probability observed across moderate-risk sites.
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <Badge variant="outline" className="border-white/10">
                            ICD drift
                          </Badge>
                          <Badge variant="outline" className="border-white/10">
                            Policy edits
                          </Badge>
                          <Badge variant="outline" className="border-white/10">
                            Appeals lag
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="rounded-2xl border-white/10 bg-black/20">
                      <CardContent className="p-4">
                        <div className="text-xs text-white/60">Recommended next</div>
                        <div className="mt-2 text-sm font-semibold">
                          Generate due diligence
                        </div>
                        <div className="mt-1 text-xs text-white/60">
                          Auto-build a partner memo with flags, forecasts, and suggested contract clauses.
                        </div>
                        <Button
                          className="mt-3 w-full rounded-2xl"
                          variant="secondary"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Create report
                        </Button>
                      </CardContent>
                    </Card>
                  </div>
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
