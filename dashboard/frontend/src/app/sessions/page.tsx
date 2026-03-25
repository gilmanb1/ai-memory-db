"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Session } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Clock, MessageSquare, FolderOpen } from "lucide-react";

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

function truncate(text: string | null, max: number): string {
  if (!text) return "";
  return text.length > max ? text.slice(0, max) + "..." : text;
}

export default function SessionsPage() {
  const [textFilter, setTextFilter] = useState("");

  const fetcher = useCallback(() => api.getSessions({ limit: "200" }), []);
  const { data } = usePolling(fetcher, 5000);

  const items: Session[] = data?.items || [];
  const filtered = textFilter
    ? items.filter(
        (s) =>
          s.trigger.toLowerCase().includes(textFilter.toLowerCase()) ||
          (s.summary || "").toLowerCase().includes(textFilter.toLowerCase()) ||
          s.cwd.toLowerCase().includes(textFilter.toLowerCase())
      )
    : items;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Sessions ({data?.total ?? 0})</h2>
      </div>

      <div className="flex gap-2 mb-4">
        <Input
          placeholder="Filter sessions..."
          value={textFilter}
          onChange={(e) => setTextFilter(e.target.value)}
          className="max-w-sm"
        />
      </div>

      <div className="space-y-3">
        {filtered.map((session) => (
          <div
            key={session.id}
            className="border rounded-lg p-4 hover:bg-muted/50 transition-colors"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">
                  {formatDate(session.created_at)}
                </span>
                <Badge variant="outline" className="text-[10px]">
                  {session.trigger}
                </Badge>
              </div>
              <div className="flex items-center gap-1 text-sm text-muted-foreground">
                <MessageSquare className="w-3.5 h-3.5" />
                <span>{session.message_count}</span>
              </div>
            </div>

            {session.summary && (
              <p className="text-sm text-muted-foreground mb-2 line-clamp-3">
                {session.summary}
              </p>
            )}

            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <FolderOpen className="w-3 h-3" />
              <span className="font-mono">{truncate(session.cwd, 60)}</span>
            </div>
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="text-center text-muted-foreground py-8">
            No sessions found.
          </div>
        )}
      </div>
    </div>
  );
}
