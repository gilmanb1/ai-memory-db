"use client";

import { useCallback } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import {
  Brain,
  Lightbulb,
  Users,
  GitBranch,
  ShieldAlert,
  ListChecks,
  Bug,
  Eye,
  Clock,
  HelpCircle,
} from "lucide-react";

const STAT_CARDS = [
  { key: "facts", label: "Facts", href: "/facts", icon: Brain, color: "text-blue-500", bg: "from-blue-500/10 to-blue-600/5" },
  { key: "decisions", label: "Decisions", href: "/decisions", icon: Lightbulb, color: "text-amber-500", bg: "from-amber-500/10 to-amber-600/5" },
  { key: "entities", label: "Entities", href: "/entities", icon: Users, color: "text-green-500", bg: "from-green-500/10 to-green-600/5" },
  { key: "relationships", label: "Relationships", href: "/relationships", icon: GitBranch, color: "text-purple-500", bg: "from-purple-500/10 to-purple-600/5" },
  { key: "guardrails", label: "Guardrails", href: "/guardrails", icon: ShieldAlert, color: "text-red-500", bg: "from-red-500/10 to-red-600/5" },
  { key: "procedures", label: "Procedures", href: "/procedures", icon: ListChecks, color: "text-cyan-500", bg: "from-cyan-500/10 to-cyan-600/5" },
  { key: "error_solutions", label: "Error Solutions", href: "/error-solutions", icon: Bug, color: "text-orange-500", bg: "from-orange-500/10 to-orange-600/5" },
  { key: "observations", label: "Observations", href: "/observations", icon: Eye, color: "text-indigo-500", bg: "from-indigo-500/10 to-indigo-600/5" },
  { key: "sessions", label: "Sessions", href: "/sessions", icon: Clock, color: "text-teal-500", bg: "from-teal-500/10 to-teal-600/5" },
  { key: "ideas", label: "Ideas", href: "/ideas", icon: Lightbulb, color: "text-pink-500", bg: "from-pink-500/10 to-pink-600/5" },
  { key: "questions", label: "Open Questions", href: "/questions", icon: HelpCircle, color: "text-slate-500", bg: "from-slate-500/10 to-slate-600/5" },
];

export default function DashboardPage() {
  const fetcher = useCallback(() => api.getStats(), []);
  const { data: stats, loading } = usePolling(fetcher, 5000);

  if (loading && !stats) {
    return <div className="text-muted-foreground p-8">Loading...</div>;
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Knowledge Base Overview</h2>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {STAT_CARDS.map(({ key, label, href, icon: Icon, color, bg }) => {
          const stat = stats?.[key];
          if (!stat) return null;
          const total = stat.total ?? 0;

          const card = (
            <Card className={`bg-gradient-to-br ${bg} border-border/50 ${href ? "cursor-pointer hover:border-border transition-colors" : ""}`}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                  <Icon className={`w-4 h-4 ${color}`} />
                  {label}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{total}</div>
                {key === "facts" && stat.long !== undefined && (
                  <div className="flex gap-1.5 mt-1.5">
                    <Badge variant="outline" className="text-[10px] border-green-500/30 text-green-400">
                      {stat.long} long
                    </Badge>
                    <Badge variant="outline" className="text-[10px] border-yellow-500/30 text-yellow-400">
                      {stat.medium} med
                    </Badge>
                    <Badge variant="outline" className="text-[10px] border-red-500/30 text-red-400">
                      {stat.short} short
                    </Badge>
                  </div>
                )}
                {key === "observations" && stat.medium !== undefined && (
                  <div className="flex gap-1.5 mt-1.5">
                    <Badge variant="outline" className="text-[10px] border-yellow-500/30 text-yellow-400">
                      {stat.medium} med
                    </Badge>
                    {stat.inactive > 0 && (
                      <Badge variant="outline" className="text-[10px] border-red-500/30 text-red-400">
                        {stat.inactive} superseded
                      </Badge>
                    )}
                  </div>
                )}
                {stat.inactive !== undefined && stat.inactive > 0 && key === "facts" && (
                  <div className="text-xs text-muted-foreground mt-1">
                    {stat.inactive} forgotten
                  </div>
                )}
              </CardContent>
            </Card>
          );

          return href ? (
            <Link key={key} href={href}>
              {card}
            </Link>
          ) : (
            <div key={key}>{card}</div>
          );
        })}
      </div>
    </div>
  );
}
