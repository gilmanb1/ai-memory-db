"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Pencil, Trash2, CheckCircle, Circle } from "lucide-react";
import { toast } from "sonner";

interface Question {
  id: string;
  text: string;
  resolved: boolean;
  scope: string;
  created_at: string;
  last_seen_at: string;
}

export default function QuestionsPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Question | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [newText, setNewText] = useState("");
  const [textFilter, setTextFilter] = useState("");
  const [showResolved, setShowResolved] = useState(false);

  const fetcher = useCallback(
    () => api.getQuestions({ limit: "200", ...(showResolved ? {} : { resolved: "false" }) }),
    [showResolved]
  );
  const { data, refetch } = usePolling(fetcher, 3000, [showResolved]);

  const items: Question[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((q) => q.text.toLowerCase().includes(textFilter.toLowerCase()))
    : items;

  async function handleCreate() {
    try {
      await api.createQuestion({ text: newText });
      toast.success("Question created");
      setCreateOpen(false);
      setNewText("");
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  async function handleToggleResolved(q: Question) {
    try {
      await api.updateQuestion(q.id, { resolved: !q.resolved });
      toast.success(q.resolved ? "Marked unresolved" : "Marked resolved");
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  async function handleUpdate() {
    if (!editItem) return;
    try {
      await api.updateQuestion(editItem.id, { text: editItem.text });
      toast.success("Question updated");
      setEditItem(null);
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteQuestion(deleteId);
      toast.success("Question deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Open Questions ({data?.total ?? 0})</h2>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add Question
        </Button>
      </div>
      <div className="flex gap-2 mb-4">
        <Input
          placeholder="Filter..."
          value={textFilter}
          onChange={(e) => setTextFilter(e.target.value)}
          className="max-w-sm"
        />
        <Button
          variant={showResolved ? "default" : "outline"}
          size="sm"
          onClick={() => setShowResolved(!showResolved)}
        >
          {showResolved ? "Showing all" : "Hiding resolved"}
        </Button>
      </div>
      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8"></TableHead>
              <TableHead className="w-[55%]">Question</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Scope</TableHead>
              <TableHead>Created</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((q) => (
              <TableRow key={q.id} className="group">
                <TableCell>
                  <button onClick={() => handleToggleResolved(q)} className="hover:opacity-80">
                    {q.resolved ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <Circle className="w-4 h-4 text-muted-foreground" />
                    )}
                  </button>
                </TableCell>
                <TableCell className={`text-sm ${q.resolved ? "line-through text-muted-foreground" : ""}`}>
                  <span className="line-clamp-2">{q.text}</span>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className={`text-[10px] ${q.resolved ? "border-green-500/30 text-green-400" : "border-yellow-500/30 text-yellow-400"}`}>
                    {q.resolved ? "resolved" : "open"}
                  </Badge>
                </TableCell>
                <TableCell className="text-xs text-muted-foreground truncate max-w-[120px]">
                  {q.scope === "__global__" ? "Global" : q.scope.split("/").pop()}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {new Date(q.created_at).toLocaleDateString()}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setEditItem({ ...q })}>
                      <Pencil className="w-3.5 h-3.5" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => setDeleteId(q.id)}>
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
          <DialogHeader><DialogTitle>Add Question</DialogTitle></DialogHeader>
          <Textarea placeholder="Question text..." value={newText} onChange={(e) => setNewText(e.target.value)} rows={3} />
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!newText.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader><DialogTitle>Edit Question</DialogTitle></DialogHeader>
          {editItem && (
            <Textarea value={editItem.text} onChange={(e) => setEditItem({ ...editItem, text: e.target.value })} rows={3} />
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditItem(null)}>Cancel</Button>
            <Button onClick={handleUpdate}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog open={!!deleteId} onOpenChange={(o) => !o && setDeleteId(null)} title="Delete Question" description="This will soft-delete the question." onConfirm={handleDelete} />
    </div>
  );
}
