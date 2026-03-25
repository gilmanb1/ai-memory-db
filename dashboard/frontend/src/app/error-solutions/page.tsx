"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { ErrorSolution } from "@/lib/types";
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

export default function ErrorSolutionsPage() {
  const [textFilter, setTextFilter] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<ErrorSolution | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Create form
  const [newErrorPattern, setNewErrorPattern] = useState("");
  const [newSolution, setNewSolution] = useState("");
  const [newErrorContext, setNewErrorContext] = useState("");
  const [newFilePaths, setNewFilePaths] = useState("");

  const fetcher = useCallback(() => api.getErrorSolutions({ limit: "200" }), []);
  const { data, refetch } = usePolling(fetcher, 3000);

  const items: ErrorSolution[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((e) =>
        e.error_pattern.toLowerCase().includes(textFilter.toLowerCase())
      )
    : items;

  async function handleCreate() {
    try {
      const filePaths = newFilePaths.trim()
        ? newFilePaths.split(",").map((p) => p.trim()).filter(Boolean)
        : null;
      await api.createErrorSolution({
        error_pattern: newErrorPattern,
        solution: newSolution,
        error_context: newErrorContext,
        file_paths: filePaths,
      });
      toast.success("Error solution created");
      setCreateOpen(false);
      setNewErrorPattern("");
      setNewSolution("");
      setNewErrorContext("");
      setNewFilePaths("");
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
      await api.updateErrorSolution(editItem.id, {
        error_pattern: editItem.error_pattern,
        solution: editItem.solution,
        error_context: editItem.error_context,
        file_paths: filePaths,
      });
      toast.success("Error solution updated");
      setEditItem(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteErrorSolution(deleteId);
      toast.success("Error solution deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Error Solutions ({data?.total ?? 0})</h2>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add Error Solution
        </Button>
      </div>

      <div className="flex gap-2 mb-4">
        <Input
          placeholder="Filter by error pattern..."
          value={textFilter}
          onChange={(e) => setTextFilter(e.target.value)}
          className="max-w-sm"
        />
      </div>

      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Error Pattern</TableHead>
              <TableHead className="w-[25%]">Solution</TableHead>
              <TableHead>Context</TableHead>
              <TableHead>File Paths</TableHead>
              <TableHead className="text-center">Confidence</TableHead>
              <TableHead className="text-center">Applied</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((item) => (
              <TableRow key={item.id} className="group">
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2 font-mono text-xs">{item.error_pattern}</span>
                </TableCell>
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2">{item.solution}</span>
                </TableCell>
                <TableCell className="text-sm max-w-xs">
                  <span className="line-clamp-2 text-muted-foreground">{item.error_context}</span>
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
                <TableCell className="text-center">
                  <Badge variant="outline" className="text-[10px]">
                    {item.confidence}
                  </Badge>
                </TableCell>
                <TableCell className="text-center text-sm text-muted-foreground">
                  {item.times_applied}
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
            <DialogTitle>Add Error Solution</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <Input
              placeholder="Error pattern..."
              value={newErrorPattern}
              onChange={(e) => setNewErrorPattern(e.target.value)}
            />
            <Textarea
              placeholder="Solution..."
              value={newSolution}
              onChange={(e) => setNewSolution(e.target.value)}
              rows={3}
            />
            <Textarea
              placeholder="Error context..."
              value={newErrorContext}
              onChange={(e) => setNewErrorContext(e.target.value)}
              rows={2}
            />
            <Input
              placeholder="File paths (comma-separated)"
              value={newFilePaths}
              onChange={(e) => setNewFilePaths(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!newErrorPattern.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Error Solution</DialogTitle>
          </DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <Input
                value={editItem.error_pattern}
                onChange={(e) =>
                  setEditItem({ ...editItem, error_pattern: e.target.value })
                }
              />
              <Textarea
                value={editItem.solution}
                onChange={(e) => setEditItem({ ...editItem, solution: e.target.value })}
                rows={3}
              />
              <Textarea
                value={editItem.error_context}
                onChange={(e) =>
                  setEditItem({ ...editItem, error_context: e.target.value })
                }
                rows={2}
              />
              <Input
                placeholder="File paths (comma-separated)"
                value={Array.isArray(editItem.file_paths) ? editItem.file_paths.join(", ") : ""}
                onChange={(e) =>
                  setEditItem({
                    ...editItem,
                    file_paths: e.target.value ? e.target.value.split(",").map((p) => p.trim()) : null,
                  } as ErrorSolution)
                }
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
        title="Delete Error Solution"
        description="This will soft-delete the error solution. It can be recovered within 30 days."
        onConfirm={handleDelete}
      />
    </div>
  );
}
