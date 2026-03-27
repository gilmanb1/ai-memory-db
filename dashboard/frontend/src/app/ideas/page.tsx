"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { useScope } from "@/context/scope-context";
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
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { DecayBar } from "@/components/decay-bar";
import { TemporalBadge } from "@/components/temporal-badge";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";

interface Idea {
  id: string;
  text: string;
  idea_type: string;
  temporal_class: string;
  decay_score: number;
  session_count: number;
  scope: string;
  created_at: string;
  last_seen_at: string;
}

const IDEA_TYPES = ["insight", "hypothesis", "suggestion", "improvement", "question"];

export default function IdeasPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Idea | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [newText, setNewText] = useState("");
  const [newType, setNewType] = useState("insight");
  const [textFilter, setTextFilter] = useState("");
  const { scopeParam, selectedScope } = useScope();

  const fetcher = useCallback(() => api.getIdeas({ ...scopeParam, limit: "200" }), [selectedScope]);
  const { data, refetch } = usePolling(fetcher, 3000, [selectedScope]);

  const items: Idea[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((i) => i.text.toLowerCase().includes(textFilter.toLowerCase()))
    : items;

  async function handleCreate() {
    try {
      await api.createIdea({ text: newText, idea_type: newType });
      toast.success("Idea created");
      setCreateOpen(false);
      setNewText("");
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  async function handleUpdate() {
    if (!editItem) return;
    try {
      await api.updateIdea(editItem.id, {
        text: editItem.text,
        idea_type: editItem.idea_type,
        temporal_class: editItem.temporal_class,
      });
      toast.success("Idea updated");
      setEditItem(null);
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteIdea(deleteId);
      toast.success("Idea deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) { toast.error(e.message); }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Ideas ({data?.total ?? 0})</h2>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4 mr-1" /> Add Idea
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
              <TableHead>Type</TableHead>
              <TableHead>Class</TableHead>
              <TableHead>Decay</TableHead>
              <TableHead className="text-center">Sessions</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((idea) => (
              <TableRow key={idea.id} className="group">
                <TableCell className="text-sm"><span className="line-clamp-2">{idea.text}</span></TableCell>
                <TableCell>
                  <Badge variant="outline" className="text-[10px]">{idea.idea_type}</Badge>
                </TableCell>
                <TableCell><TemporalBadge value={idea.temporal_class} /></TableCell>
                <TableCell><DecayBar score={idea.decay_score} /></TableCell>
                <TableCell className="text-center text-sm text-muted-foreground">{idea.session_count}</TableCell>
                <TableCell>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setEditItem({ ...idea })}>
                      <Pencil className="w-3.5 h-3.5" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => setDeleteId(idea.id)}>
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
          <DialogHeader><DialogTitle>Add Idea</DialogTitle></DialogHeader>
          <Textarea placeholder="Idea text..." value={newText} onChange={(e) => setNewText(e.target.value)} rows={3} />
          <Select value={newType} onValueChange={(v) => v && setNewType(v)}>
            <SelectTrigger><SelectValue placeholder="Type" /></SelectTrigger>
            <SelectContent>
              {IDEA_TYPES.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
            </SelectContent>
          </Select>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!newText.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader><DialogTitle>Edit Idea</DialogTitle></DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <Textarea value={editItem.text} onChange={(e) => setEditItem({ ...editItem, text: e.target.value })} rows={3} />
              <div className="flex gap-2">
                <Select value={editItem.idea_type} onValueChange={(v) => v && setEditItem({ ...editItem, idea_type: v })}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {IDEA_TYPES.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                  </SelectContent>
                </Select>
                <Select value={editItem.temporal_class} onValueChange={(v) => v && setEditItem({ ...editItem, temporal_class: v })}>
                  <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="long">Long</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="short">Short</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditItem(null)}>Cancel</Button>
            <Button onClick={handleUpdate}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog open={!!deleteId} onOpenChange={(o) => !o && setDeleteId(null)} title="Delete Idea" description="This will soft-delete the idea." onConfirm={handleDelete} />
    </div>
  );
}
