"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Observation } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { DecayBar } from "@/components/decay-bar";
import { TemporalBadge } from "@/components/temporal-badge";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Trash2 } from "lucide-react";
import { toast } from "sonner";

export default function ObservationsPage() {
  const [textFilter, setTextFilter] = useState("");
  const [deleteId, setDeleteId] = useState<string | null>(null);

  const fetcher = useCallback(() => api.getObservations({ limit: "200" }), []);
  const { data, refetch } = usePolling(fetcher, 3000);

  const items: Observation[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((o) => o.text.toLowerCase().includes(textFilter.toLowerCase()))
    : items;

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteObservation(deleteId);
      toast.success("Observation deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Observations ({data?.total ?? 0})</h2>
      </div>

      <div className="flex gap-2 mb-4">
        <Input
          placeholder="Filter by text..."
          value={textFilter}
          onChange={(e) => setTextFilter(e.target.value)}
          className="max-w-sm"
        />
      </div>

      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[40%]">Text</TableHead>
              <TableHead className="text-center">Proofs</TableHead>
              <TableHead>Class</TableHead>
              <TableHead>Decay</TableHead>
              <TableHead className="text-center">Imp.</TableHead>
              <TableHead className="text-center">Source Facts</TableHead>
              <TableHead className="w-12"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((item) => (
              <TableRow key={item.id} className="group">
                <TableCell className="text-sm max-w-md">
                  <span className="line-clamp-2">{item.text}</span>
                </TableCell>
                <TableCell className="text-center text-sm">{item.proof_count}</TableCell>
                <TableCell>
                  <TemporalBadge value={item.temporal_class} />
                </TableCell>
                <TableCell>
                  <DecayBar score={item.decay_score} />
                </TableCell>
                <TableCell className="text-center text-sm">{item.importance}</TableCell>
                <TableCell className="text-center text-sm text-muted-foreground">
                  {item.source_fact_ids?.length ?? 0}
                </TableCell>
                <TableCell>
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-destructive"
                      onClick={() => setDeleteId(item.id)}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Delete Confirm */}
      <ConfirmDialog
        open={!!deleteId}
        onOpenChange={(o) => !o && setDeleteId(null)}
        title="Delete Observation"
        description="This will soft-delete the observation. It can be recovered within 30 days."
        onConfirm={handleDelete}
      />
    </div>
  );
}
