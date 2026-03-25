"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const TEMPORAL_STYLES: Record<string, string> = {
  long: "border-green-500/30 text-green-400 bg-green-500/10",
  medium: "border-yellow-500/30 text-yellow-400 bg-yellow-500/10",
  short: "border-red-500/30 text-red-400 bg-red-500/10",
};

export function TemporalBadge({ value }: { value: string }) {
  return (
    <Badge variant="outline" className={cn("text-[10px]", TEMPORAL_STYLES[value] || "")}>
      {value}
    </Badge>
  );
}
