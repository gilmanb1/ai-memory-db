"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Fact } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DecayBar } from "@/components/decay-bar";
import { TemporalBadge } from "@/components/temporal-badge";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Pencil, Trash2, TrendingUp, TrendingDown } from "lucide-react";
import { toast } from "sonner";
import { useScope } from "@/context/scope-context";

const CATEGORIES = [
  "architecture", "implementation", "operational", "dependency",
  "decision_rationale", "constraint", "bug_pattern", "user_preference",
  "project_context", "technical", "decision", "personal", "contextual", "numerical",
];

export default function FactsPage() {
  const [filterClass, setFilterClass] = useState<string>("all");
  const [textFilter, setTextFilter] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Fact | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Create form
  const [newText, setNewText] = useState("");
  const [newCategory, setNewCategory] = useState("contextual");
  const [newClass, setNewClass] = useState("long");
  const [newImportance, setNewImportance] = useState("5");

  const { scopeParam, selectedScope } = useScope();
  const params: Record<string, string> = { limit: "200", ...scopeParam };
  if (filterClass !== "all") params.temporal_class = filterClass;

  const fetcher = useCallback(() => api.getFacts(params), [filterClass, selectedScope]);
  const { data, refetch } = usePolling(fetcher, 3000, [filterClass]);

  const items: Fact[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((f) => f.text.toLowerCase().includes(textFilter.toLowerCase()))
    : items;

  async function handleCreate() {
    try {
      await api.createFact({
        text: newText,
        category: newCategory,
        temporal_class: newClass,
        importance: parseInt(newImportance),
      });
      toast.success("Fact created");
      setCreateOpen(false);
      setNewText("");
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleUpdate() {
    if (!editItem) return;
    try {
      await api.updateFact(editItem.id, {
        text: editItem.text,
        category: editItem.category,
        temporal_class: editItem.temporal_class,
        importance: editItem.importance,
      });
      toast.success("Fact updated");
      setEditItem(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteFact(deleteId);
      toast.success("Fact deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Facts ({data?.total ?? 0})</h2>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add Fact
        </Button>
      </div>

      <div className="flex gap-2 mb-4">
        <Input
          placeholder="Filter by text..."
          value={textFilter}
          onChange={(e) => setTextFilter(e.target.value)}
          className="max-w-sm"
        />
        <Select value={filterClass} onValueChange={(v) => v && setFilterClass(v)}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All</SelectItem>
            <SelectItem value="long">Long</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="short">Short</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[40%]">Text</TableHead>
              <TableHead>Category</TableHead>
              <TableHead>Class</TableHead>
              <TableHead>Decay</TableHead>
              <TableHead className="text-center">Imp.</TableHead>
              <TableHead className="text-center">Recalled</TableHead>
              <TableHead>Trend</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((fact) => (
              <TableRow key={fact.id} className="group">
                <TableCell className="text-sm max-w-md">
                  <span className="line-clamp-2">{fact.text}</span>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className="text-[10px]">
                    {fact.category}
                  </Badge>
                </TableCell>
                <TableCell>
                  <TemporalBadge value={fact.temporal_class} />
                </TableCell>
                <TableCell>
                  <DecayBar score={fact.decay_score} />
                </TableCell>
                <TableCell className="text-center text-sm">{fact.importance}</TableCell>
                <TableCell className="text-center text-sm text-muted-foreground">
                  {fact.times_recalled}
                </TableCell>
                <TableCell>
                  {fact.decay_score >= 0.95 ? (
                    <TrendingUp className="w-4 h-4 text-green-500" />
                  ) : fact.decay_score < 0.5 ? (
                    <TrendingDown className="w-4 h-4 text-red-500" />
                  ) : null}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => setEditItem({ ...fact })}
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-destructive"
                      onClick={() => setDeleteId(fact.id)}
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

      {/* Create Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Fact</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <Textarea
              placeholder="Fact text..."
              value={newText}
              onChange={(e) => setNewText(e.target.value)}
              rows={3}
            />
            <div className="flex gap-2">
              <Select value={newCategory} onValueChange={(v) => v && setNewCategory(v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent>
                  {CATEGORIES.map((c) => (
                    <SelectItem key={c} value={c}>{c}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={newClass} onValueChange={(v) => v && setNewClass(v)}>
                <SelectTrigger className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="long">Long</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="short">Short</SelectItem>
                </SelectContent>
              </Select>
              <Input
                type="number"
                placeholder="Importance"
                value={newImportance}
                onChange={(e) => setNewImportance(e.target.value)}
                className="w-20"
                min={1}
                max={10}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!newText.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Fact</DialogTitle>
          </DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <Textarea
                value={editItem.text}
                onChange={(e) => setEditItem({ ...editItem, text: e.target.value })}
                rows={3}
              />
              <div className="flex gap-2">
                <Select
                  value={editItem.category}
                  onValueChange={(v) => v && setEditItem({ ...editItem, category: v })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {CATEGORIES.map((c) => (
                      <SelectItem key={c} value={c}>{c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select
                  value={editItem.temporal_class}
                  onValueChange={(v) => v && setEditItem({ ...editItem, temporal_class: v })}
                >
                  <SelectTrigger className="w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="long">Long</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="short">Short</SelectItem>
                  </SelectContent>
                </Select>
                <Input
                  type="number"
                  value={editItem.importance}
                  onChange={(e) =>
                    setEditItem({ ...editItem, importance: parseInt(e.target.value) || 5 })
                  }
                  className="w-20"
                  min={1}
                  max={10}
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditItem(null)}>Cancel</Button>
            <Button onClick={handleUpdate}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirm */}
      <ConfirmDialog
        open={!!deleteId}
        onOpenChange={(o) => !o && setDeleteId(null)}
        title="Delete Fact"
        description="This will soft-delete the fact. It can be recovered within 30 days."
        onConfirm={handleDelete}
      />
    </div>
  );
}
