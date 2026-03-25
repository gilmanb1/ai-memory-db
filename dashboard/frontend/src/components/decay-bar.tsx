"use client";

import { cn } from "@/lib/utils";

export function DecayBar({ score, className }: { score: number; className?: string }) {
  const pct = Math.max(0, Math.min(100, score * 100));
  const color =
    pct >= 80 ? "bg-green-500" : pct >= 50 ? "bg-yellow-500" : pct >= 20 ? "bg-orange-500" : "bg-red-500";

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-muted-foreground w-8">{score.toFixed(2)}</span>
    </div>
  );
}
