"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useScope } from "@/context/scope-context";
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
  Search,
  LayoutDashboard,
  HelpCircle,
  Code2,
  Waypoints,
  ClipboardCheck,
  FolderOpen,
  BookOpen,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/facts", label: "Facts", icon: Brain },
  { href: "/decisions", label: "Decisions", icon: Lightbulb },
  { href: "/entities", label: "Entities", icon: Users },
  { href: "/relationships", label: "Relationships", icon: GitBranch },
  { href: "/code-graph", label: "Code Graph", icon: Code2 },
  { href: "/knowledge", label: "Knowledge Graph", icon: Waypoints },
  { href: "/guardrails", label: "Guardrails", icon: ShieldAlert },
  { href: "/procedures", label: "Procedures", icon: ListChecks },
  { href: "/error-solutions", label: "Error Solutions", icon: Bug },
  { href: "/ideas", label: "Ideas", icon: Lightbulb },
  { href: "/observations", label: "Observations", icon: Eye },
  { href: "/sessions", label: "Sessions", icon: Clock },
  { href: "/questions", label: "Open Questions", icon: HelpCircle },
  { href: "/search", label: "Search", icon: Search },
  { href: "/review", label: "Review Queue", icon: ClipboardCheck },
  { href: "/api-docs", label: "API Docs", icon: BookOpen },
];

export function Sidebar() {
  const pathname = usePathname();
  const { selectedScope, setSelectedScope, scopes } = useScope();

  return (
    <aside className="fixed left-0 top-0 h-full w-56 bg-card border-r border-border flex flex-col z-50">
      <div className="p-4 border-b border-border">
        <h1 className="text-sm font-semibold tracking-tight">
          <Brain className="inline-block w-4 h-4 mr-1.5 -mt-0.5 text-blue-500" />
          Memory Dashboard
        </h1>
      </div>

      {/* Scope selector */}
      {scopes.length > 1 && (
        <div className="px-3 py-2 border-b border-border">
          <div className="flex items-center gap-1.5 mb-1.5">
            <FolderOpen className="w-3 h-3 text-muted-foreground" />
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">Scope</span>
          </div>
          <select
            value={selectedScope || "__all__"}
            onChange={(e) => setSelectedScope(e.target.value === "__all__" ? null : e.target.value)}
            className="w-full text-xs bg-background border border-border rounded px-2 py-1.5 text-foreground focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="__all__">All scopes</option>
            {scopes.map((s) => (
              <option key={s.scope} value={s.scope}>
                {s.display_name} ({s.count})
              </option>
            ))}
          </select>
        </div>
      )}

      <nav className="flex-1 overflow-y-auto py-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 px-4 py-2 text-sm transition-colors",
                active
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-3 border-t border-border text-xs text-muted-foreground">
        {selectedScope ? scopes.find(s => s.scope === selectedScope)?.display_name || "filtered" : "all scopes"}
      </div>
    </aside>
  );
}
