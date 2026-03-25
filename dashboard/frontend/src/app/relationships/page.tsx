"use client";

import { useCallback, useState } from "react";
import dynamic from "next/dynamic";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Relationship } from "@/lib/types";
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
import { Plus, Pencil, Trash2, List, Network } from "lucide-react";
import { toast } from "sonner";

const EntityGraph = dynamic(() => import("@/components/entity-graph").then((m) => ({ default: m.EntityGraph })), { ssr: false });

export default function RelationshipsPage() {
  const [view, setView] = useState<"list" | "graph">("list");
  const [textFilter, setTextFilter] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Relationship | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Create form
  const [newFromEntity, setNewFromEntity] = useState("");
  const [newToEntity, setNewToEntity] = useState("");
  const [newRelType, setNewRelType] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newStrength, setNewStrength] = useState("5");

  const fetcher = useCallback(() => api.getRelationships({ limit: "200" }), []);
  const { data, refetch } = usePolling(fetcher, 3000);

  const items: Relationship[] = data?.items || [];
  const filtered = textFilter
    ? items.filter(
        (r) =>
          r.from_entity.toLowerCase().includes(textFilter.toLowerCase()) ||
          r.to_entity.toLowerCase().includes(textFilter.toLowerCase()) ||
          r.rel_type.toLowerCase().includes(textFilter.toLowerCase()) ||
          r.description.toLowerCase().includes(textFilter.toLowerCase())
      )
    : items;

  async function handleCreate() {
    try {
      await api.createRelationship({
        from_entity: newFromEntity,
        to_entity: newToEntity,
        rel_type: newRelType,
        description: newDescription,
        strength: parseInt(newStrength),
      });
      toast.success("Relationship created");
      setCreateOpen(false);
      setNewFromEntity("");
      setNewToEntity("");
      setNewRelType("");
      setNewDescription("");
      setNewStrength("5");
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleUpdate() {
    if (!editItem) return;
    try {
      await api.updateRelationship(editItem.id, {
        description: editItem.description,
        rel_type: editItem.rel_type,
        strength: editItem.strength,
      });
      toast.success("Relationship updated");
      setEditItem(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteRelationship(deleteId);
      toast.success("Relationship deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Relationships ({data?.total ?? 0})</h2>
        <div className="flex gap-2">
          <div className="flex border rounded-md overflow-hidden">
            <button
              onClick={() => setView("list")}
              className={`flex items-center gap-1 px-3 py-1.5 text-sm transition-colors ${
                view === "list"
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50"
              }`}
            >
              <List className="w-4 h-4" /> List
            </button>
            <button
              onClick={() => setView("graph")}
              className={`flex items-center gap-1 px-3 py-1.5 text-sm transition-colors ${
                view === "graph"
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50"
              }`}
            >
              <Network className="w-4 h-4" /> Graph
            </button>
          </div>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4 mr-1" /> Add
          </Button>
        </div>
      </div>

      {view === "graph" ? (
        <div className="border rounded-md overflow-hidden" style={{ height: "calc(100vh - 140px)" }}>
          <EntityGraph />
        </div>
      ) : (
        <>
          <div className="flex gap-2 mb-4">
            <Input
              placeholder="Filter by entity or type..."
              value={textFilter}
              onChange={(e) => setTextFilter(e.target.value)}
              className="max-w-sm"
            />
          </div>

          <div className="border rounded-md">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>From</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>To</TableHead>
                  <TableHead className="w-[30%]">Description</TableHead>
                  <TableHead className="text-center">Strength</TableHead>
                  <TableHead className="text-center">Sessions</TableHead>
                  <TableHead className="w-16"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((item) => (
                  <TableRow key={item.id} className="group">
                    <TableCell className="text-sm font-medium">{item.from_entity}</TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="text-[10px]">
                        {item.rel_type}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm font-medium">{item.to_entity}</TableCell>
                    <TableCell className="text-sm max-w-xs">
                      <span className="line-clamp-2 text-muted-foreground">{item.description}</span>
                    </TableCell>
                    <TableCell className="text-center text-sm">{item.strength}</TableCell>
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
        </>
      )}

      {/* Create Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Relationship</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div className="flex gap-2">
              <Input
                placeholder="From entity..."
                value={newFromEntity}
                onChange={(e) => setNewFromEntity(e.target.value)}
              />
              <Input
                placeholder="To entity..."
                value={newToEntity}
                onChange={(e) => setNewToEntity(e.target.value)}
              />
            </div>
            <Input
              placeholder="Relationship type..."
              value={newRelType}
              onChange={(e) => setNewRelType(e.target.value)}
            />
            <Textarea
              placeholder="Description..."
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              rows={2}
            />
            <Input
              type="number"
              placeholder="Strength"
              value={newStrength}
              onChange={(e) => setNewStrength(e.target.value)}
              className="w-20"
              min={1}
              max={10}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button
              onClick={handleCreate}
              disabled={!newFromEntity.trim() || !newToEntity.trim() || !newRelType.trim()}
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Relationship</DialogTitle>
          </DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <div className="flex gap-2">
                <Input value={editItem.from_entity} disabled className="opacity-60" />
                <Input value={editItem.to_entity} disabled className="opacity-60" />
              </div>
              <Input
                placeholder="Relationship type..."
                value={editItem.rel_type}
                onChange={(e) => setEditItem({ ...editItem, rel_type: e.target.value })}
              />
              <Textarea
                value={editItem.description}
                onChange={(e) => setEditItem({ ...editItem, description: e.target.value })}
                rows={2}
              />
              <Input
                type="number"
                value={editItem.strength}
                onChange={(e) => setEditItem({ ...editItem, strength: parseInt(e.target.value) || 5 })}
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
        title="Delete Relationship"
        description="This will soft-delete the relationship. It can be recovered within 30 days."
        onConfirm={handleDelete}
      />
    </div>
  );
}
