"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
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
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/facts", label: "Facts", icon: Brain },
  { href: "/decisions", label: "Decisions", icon: Lightbulb },
  { href: "/entities", label: "Entities", icon: Users },
  { href: "/relationships", label: "Relationships", icon: GitBranch },
  { href: "/guardrails", label: "Guardrails", icon: ShieldAlert },
  { href: "/procedures", label: "Procedures", icon: ListChecks },
  { href: "/error-solutions", label: "Error Solutions", icon: Bug },
  { href: "/ideas", label: "Ideas", icon: Lightbulb },
  { href: "/observations", label: "Observations", icon: Eye },
  { href: "/sessions", label: "Sessions", icon: Clock },
  { href: "/questions", label: "Open Questions", icon: HelpCircle },
  { href: "/search", label: "Search", icon: Search },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-56 bg-card border-r border-border flex flex-col z-50">
      <div className="p-4 border-b border-border">
        <h1 className="text-sm font-semibold tracking-tight">
          <Brain className="inline-block w-4 h-4 mr-1.5 -mt-0.5 text-blue-500" />
          Memory Dashboard
        </h1>
      </div>
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
        ai-memory-db
      </div>
    </aside>
  );
}
