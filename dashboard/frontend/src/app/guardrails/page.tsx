"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { useScope } from "@/context/scope-context";
import { Guardrail } from "@/lib/types";
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
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";

export default function GuardrailsPage() {
  const [textFilter, setTextFilter] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Guardrail | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Create form
  const [newWarning, setNewWarning] = useState("");
  const [newRationale, setNewRationale] = useState("");
  const [newConsequence, setNewConsequence] = useState("");
  const [newFilePaths, setNewFilePaths] = useState("");
  const [newImportance, setNewImportance] = useState("5");
  const { scopeParam, selectedScope } = useScope();

  const fetcher = useCallback(() => api.getGuardrails({ ...scopeParam, limit: "200" }), [selectedScope]);
  const { data, refetch } = usePolling(fetcher, 3000, [selectedScope]);

  const items: Guardrail[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((g) => g.warning.toLowerCase().includes(textFilter.toLowerCase()))
    : items;

  async function handleCreate() {
    try {
      const filePaths = newFilePaths.trim()
        ? newFilePaths.split(",").map((p) => p.trim()).filter(Boolean)
        : null;
      await api.createGuardrail({
        warning: newWarning,
        rationale: newRationale,
        consequence: newConsequence,
        file_paths: filePaths,
        importance: parseInt(newImportance),
      });
      toast.success("Guardrail created");
      setCreateOpen(false);
      setNewWarning("");
      setNewRationale("");
      setNewConsequence("");
      setNewFilePaths("");
      setNewImportance("5");
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleUpdate() {
    if (!editItem) return;
    try {
      const filePaths =
        typeof editItem.file_paths === "string"
          ? (editItem.file_paths as string).split(",").map((p: string) => p.trim()).filter(Boolean)
          : editItem.file_paths;
      await api.updateGuardrail(editItem.id, {
        warning: editItem.warning,
        rationale: editItem.rationale,
        consequence: editItem.consequence,
        file_paths: filePaths,
        importance: editItem.importance,
      });
      toast.success("Guardrail updated");
      setEditItem(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteGuardrail(deleteId);
      toast.success("Guardrail deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Guardrails ({data?.total ?? 0})</h2>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add Guardrail
        </Button>
      </div>

      <div className="flex gap-2 mb-4">
        <Input
          placeholder="Filter by warning..."
          value={textFilter}
          onChange={(e) => setTextFilter(e.target.value)}
          className="max-w-sm"
        />
      </div>

      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[30%]">Warning</TableHead>
              <TableHead>Rationale</TableHead>
              <TableHead>Consequence</TableHead>
              <TableHead>File Paths</TableHead>
              <TableHead className="text-center">Imp.</TableHead>
              <TableHead className="text-center">Sessions</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((item) => (
              <TableRow key={item.id} className="group">
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2">{item.warning}</span>
                </TableCell>
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2 text-muted-foreground">{item.rationale}</span>
                </TableCell>
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2 text-muted-foreground">{item.consequence}</span>
                </TableCell>
                <TableCell>
                  <div className="flex flex-wrap gap-1">
                    {item.file_paths?.map((fp, i) => (
                      <Badge key={i} variant="outline" className="text-[10px]">
                        {fp}
                      </Badge>
                    ))}
                  </div>
                </TableCell>
                <TableCell className="text-center text-sm">{item.importance}</TableCell>
                <TableCell className="text-center text-sm text-muted-foreground">
                  {item.session_count}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => setEditItem({ ...item })}
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </Button>
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

      {/* Create Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Guardrail</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <Textarea
              placeholder="Warning..."
              value={newWarning}
              onChange={(e) => setNewWarning(e.target.value)}
              rows={2}
            />
            <Textarea
              placeholder="Rationale..."
              value={newRationale}
              onChange={(e) => setNewRationale(e.target.value)}
              rows={2}
            />
            <Textarea
              placeholder="Consequence..."
              value={newConsequence}
              onChange={(e) => setNewConsequence(e.target.value)}
              rows={2}
            />
            <Input
              placeholder="File paths (comma-separated)"
              value={newFilePaths}
              onChange={(e) => setNewFilePaths(e.target.value)}
            />
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
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!newWarning.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Guardrail</DialogTitle>
          </DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <Textarea
                value={editItem.warning}
                onChange={(e) => setEditItem({ ...editItem, warning: e.target.value })}
                rows={2}
              />
              <Textarea
                value={editItem.rationale}
                onChange={(e) => setEditItem({ ...editItem, rationale: e.target.value })}
                rows={2}
              />
              <Textarea
                value={editItem.consequence}
                onChange={(e) => setEditItem({ ...editItem, consequence: e.target.value })}
                rows={2}
              />
              <Input
                placeholder="File paths (comma-separated)"
                value={Array.isArray(editItem.file_paths) ? editItem.file_paths.join(", ") : ""}
                onChange={(e) =>
                  setEditItem({
                    ...editItem,
                    file_paths: e.target.value ? e.target.value.split(",").map((p) => p.trim()) : null,
                  } as Guardrail)
                }
              />
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
        title="Delete Guardrail"
        description="This will soft-delete the guardrail. It can be recovered within 30 days."
        onConfirm={handleDelete}
      />
    </div>
  );
}
