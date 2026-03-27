"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { useScope } from "@/context/scope-context";
import { Decision } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { DecayBar } from "@/components/decay-bar";
import { TemporalBadge } from "@/components/temporal-badge";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";

export default function DecisionsPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Decision | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [newText, setNewText] = useState("");
  const [newImportance, setNewImportance] = useState("7");
  const [textFilter, setTextFilter] = useState("");
  const { scopeParam, selectedScope } = useScope();

  const fetcher = useCallback(() => api.getDecisions({ ...scopeParam, limit: "200" }), [selectedScope]);
  const { data, refetch } = usePolling(fetcher, 3000, [selectedScope]);

  const items: Decision[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((d) => d.text.toLowerCase().includes(textFilter.toLowerCase()))
    : items;

  async function handleCreate() {
    try {
      await api.createDecision({ text: newText, importance: parseInt(newImportance) });
      toast.success("Decision created");
      setCreateOpen(false);
      setNewText("");
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  async function handleUpdate() {
    if (!editItem) return;
    try {
      await api.updateDecision(editItem.id, {
        text: editItem.text,
        temporal_class: editItem.temporal_class,
        importance: editItem.importance,
      });
      toast.success("Decision updated");
      setEditItem(null);
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteDecision(deleteId);
      toast.success("Decision deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Decisions ({data?.total ?? 0})</h2>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add Decision
        </Button>
      </div>
      <Input
        placeholder="Filter..."
        value={textFilter}
        onChange={(e) => setTextFilter(e.target.value)}
        className="max-w-sm mb-4"
      />
      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[50%]">Text</TableHead>
              <TableHead>Class</TableHead>
              <TableHead>Decay</TableHead>
              <TableHead className="text-center">Imp.</TableHead>
              <TableHead className="text-center">Sessions</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((d) => (
              <TableRow key={d.id} className="group">
                <TableCell className="text-sm"><span className="line-clamp-2">{d.text}</span></TableCell>
                <TableCell><TemporalBadge value={d.temporal_class} /></TableCell>
                <TableCell><DecayBar score={d.decay_score} /></TableCell>
                <TableCell className="text-center text-sm">{d.importance}</TableCell>
                <TableCell className="text-center text-sm text-muted-foreground">{d.session_count}</TableCell>
                <TableCell>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setEditItem({ ...d })}>
                      <Pencil className="w-3.5 h-3.5" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => setDeleteId(d.id)}>
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Add Decision</DialogTitle></DialogHeader>
          <Textarea placeholder="Decision text..." value={newText} onChange={(e) => setNewText(e.target.value)} rows={3} />
          <Input type="number" placeholder="Importance (1-10)" value={newImportance} onChange={(e) => setNewImportance(e.target.value)} className="w-28" min={1} max={10} />
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!newText.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader><DialogTitle>Edit Decision</DialogTitle></DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <Textarea value={editItem.text} onChange={(e) => setEditItem({ ...editItem, text: e.target.value })} rows={3} />
              <div className="flex gap-2">
                <Select value={editItem.temporal_class} onValueChange={(v) => v && setEditItem({ ...editItem, temporal_class: v })}>
                  <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="long">Long</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="short">Short</SelectItem>
                  </SelectContent>
                </Select>
                <Input type="number" value={editItem.importance} onChange={(e) => setEditItem({ ...editItem, importance: parseInt(e.target.value) || 7 })} className="w-20" min={1} max={10} />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditItem(null)}>Cancel</Button>
            <Button onClick={handleUpdate}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog open={!!deleteId} onOpenChange={(o) => !o && setDeleteId(null)} title="Delete Decision" description="This will soft-delete the decision." onConfirm={handleDelete} />
    </div>
  );
}
