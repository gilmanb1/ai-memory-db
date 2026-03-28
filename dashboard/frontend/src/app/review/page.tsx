"use client";

import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle, Clock, AlertTriangle } from "lucide-react";

interface ReviewItem {
  id: string;
  item_text: string;
  item_table: string;
  item_data: string;
  reason: string;
  status: string;
  source_session: string;
  scope: string;
  created_at: string;
  reviewed_at: string | null;
}

const REASON_LABELS: Record<string, string> = {
  low_confidence_high_importance: "Low confidence, high importance",
  uncertainty_markers: "Contains uncertainty language",
  low_confidence_low_importance: "Low confidence",
  pattern_match: "Pattern rejection",
};

const TABLE_COLORS: Record<string, string> = {
  facts: "bg-green-500/20 text-green-400",
  decisions: "bg-purple-500/20 text-purple-400",
  guardrails: "bg-red-500/20 text-red-400",
  procedures: "bg-cyan-500/20 text-cyan-400",
};

export default function ReviewPage() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [total, setTotal] = useState(0);
  const [filter, setFilter] = useState<"pending" | "approved" | "rejected">("pending");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ status: filter, limit: "100" });
      const res = await fetch(`${(() => { const u = process.env.NEXT_PUBLIC_API_URL; return (!u || u === "__SAME_ORIGIN__") ? "" : u; })()}/api/v1/review?${params}`);
      const data = await res.json();
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  async function handleAction(id: string, action: "approve" | "reject") {
    setActionLoading(id);
    try {
      const url = `${(() => { const u = process.env.NEXT_PUBLIC_API_URL; return (!u || u === "__SAME_ORIGIN__") ? "" : u; })()}/api/v1/review/${id}/${action}`;
      await fetch(url, { method: "POST" });
      await fetchItems();
    } finally {
      setActionLoading(null);
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Review Queue</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Flagged items from extraction that need human review
          </p>
        </div>
        <div className="flex gap-1">
          {(["pending", "approved", "rejected"] as const).map((status) => (
            <Button
              key={status}
              variant={filter === status ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setFilter(status)}
              className="capitalize"
            >
              {status === "pending" && <Clock className="w-3.5 h-3.5 mr-1" />}
              {status === "approved" && <CheckCircle className="w-3.5 h-3.5 mr-1" />}
              {status === "rejected" && <XCircle className="w-3.5 h-3.5 mr-1" />}
              {status}
            </Button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="text-center text-muted-foreground py-12">Loading...</div>
      ) : items.length === 0 ? (
        <div className="text-center py-12">
          <AlertTriangle className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
          <p className="text-muted-foreground">No {filter} items</p>
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">{total} {filter} items</p>
          {items.map((item) => (
            <div
              key={item.id}
              className="border border-border rounded-lg p-4 bg-card hover:bg-card/80 transition-colors"
            >
              <div className="flex items-start gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className={`text-xs px-1.5 py-0.5 rounded ${TABLE_COLORS[item.item_table] || "bg-gray-500/20 text-gray-400"}`}>
                      {item.item_table}
                    </span>
                    <span className="text-xs text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">
                      {REASON_LABELS[item.reason] || item.reason}
                    </span>
                    <span className="text-[10px] text-muted-foreground ml-auto">
                      {item.created_at ? new Date(item.created_at).toLocaleDateString() : ""}
                    </span>
                  </div>
                  <p className="text-sm text-foreground leading-relaxed">{item.item_text}</p>
                  <div className="flex items-center gap-3 mt-2 text-[10px] text-muted-foreground">
                    <span>ID: {item.id.slice(0, 12)}...</span>
                    {item.scope && item.scope !== "__global__" && (
                      <span>Scope: {item.scope.split("/").pop()}</span>
                    )}
                    {item.source_session && (
                      <span>Session: {item.source_session.slice(0, 12)}</span>
                    )}
                  </div>
                </div>

                {filter === "pending" && (
                  <div className="flex flex-col gap-1.5 shrink-0">
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-green-400 border-green-500/30 hover:bg-green-500/10"
                      disabled={actionLoading === item.id}
                      onClick={() => handleAction(item.id, "approve")}
                    >
                      <CheckCircle className="w-3.5 h-3.5 mr-1" />
                      Approve
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-red-400 border-red-500/30 hover:bg-red-500/10"
                      disabled={actionLoading === item.id}
                      onClick={() => handleAction(item.id, "reject")}
                    >
                      <XCircle className="w-3.5 h-3.5 mr-1" />
                      Reject
                    </Button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
