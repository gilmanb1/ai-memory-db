"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Procedure } from "@/lib/types";
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
import { DecayBar } from "@/components/decay-bar";
import { TemporalBadge } from "@/components/temporal-badge";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";

export default function ProceduresPage() {
  const [textFilter, setTextFilter] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Procedure | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Create form
  const [newTaskDescription, setNewTaskDescription] = useState("");
  const [newSteps, setNewSteps] = useState("");
  const [newFilePaths, setNewFilePaths] = useState("");
  const [newImportance, setNewImportance] = useState("5");

  const fetcher = useCallback(() => api.getProcedures({ limit: "200" }), []);
  const { data, refetch } = usePolling(fetcher, 3000);

  const items: Procedure[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((p) =>
        p.task_description.toLowerCase().includes(textFilter.toLowerCase())
      )
    : items;

  async function handleCreate() {
    try {
      const filePaths = newFilePaths.trim()
        ? newFilePaths.split(",").map((p) => p.trim()).filter(Boolean)
        : null;
      await api.createProcedure({
        task_description: newTaskDescription,
        steps: newSteps,
        file_paths: filePaths,
        importance: parseInt(newImportance),
      });
      toast.success("Procedure created");
      setCreateOpen(false);
      setNewTaskDescription("");
      setNewSteps("");
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
      await api.updateProcedure(editItem.id, {
        task_description: editItem.task_description,
        steps: editItem.steps,
        file_paths: filePaths,
        importance: editItem.importance,
      });
      toast.success("Procedure updated");
      setEditItem(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteProcedure(deleteId);
      toast.success("Procedure deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Procedures ({data?.total ?? 0})</h2>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add Procedure
        </Button>
      </div>

      <div className="flex gap-2 mb-4">
        <Input
          placeholder="Filter by task..."
          value={textFilter}
          onChange={(e) => setTextFilter(e.target.value)}
          className="max-w-sm"
        />
      </div>

      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[30%]">Task Description</TableHead>
              <TableHead>Steps</TableHead>
              <TableHead>File Paths</TableHead>
              <TableHead className="text-center">Imp.</TableHead>
              <TableHead>Class</TableHead>
              <TableHead>Decay</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((item) => (
              <TableRow key={item.id} className="group">
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2">{item.task_description}</span>
                </TableCell>
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2 text-muted-foreground">{item.steps}</span>
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
                <TableCell>
                  <TemporalBadge value={item.temporal_class} />
                </TableCell>
                <TableCell>
                  <DecayBar score={item.decay_score} />
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
            <DialogTitle>Add Procedure</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <Input
              placeholder="Task description..."
              value={newTaskDescription}
              onChange={(e) => setNewTaskDescription(e.target.value)}
            />
            <Textarea
              placeholder="Steps..."
              value={newSteps}
              onChange={(e) => setNewSteps(e.target.value)}
              rows={5}
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
            <Button onClick={handleCreate} disabled={!newTaskDescription.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Procedure</DialogTitle>
          </DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <Input
                value={editItem.task_description}
                onChange={(e) =>
                  setEditItem({ ...editItem, task_description: e.target.value })
                }
              />
              <Textarea
                value={editItem.steps}
                onChange={(e) => setEditItem({ ...editItem, steps: e.target.value })}
                rows={5}
              />
              <Input
                placeholder="File paths (comma-separated)"
                value={Array.isArray(editItem.file_paths) ? editItem.file_paths.join(", ") : ""}
                onChange={(e) =>
                  setEditItem({
                    ...editItem,
                    file_paths: e.target.value ? e.target.value.split(",").map((p) => p.trim()) : null,
                  } as Procedure)
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
        title="Delete Procedure"
        description="This will soft-delete the procedure. It can be recovered within 30 days."
        onConfirm={handleDelete}
      />
    </div>
  );
}
