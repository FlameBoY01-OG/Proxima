import { Sparkles, Trash2 } from "lucide-react";
import { Button } from "./ui";

interface Props {
  onSeed: () => void;
  onReset: () => void;
  busy: boolean;
}

export function DataControls({ onSeed, onReset, busy }: Props) {
  return (
    <div className="flex gap-2">
      <Button onClick={onSeed} disabled={busy} className="flex-1 justify-center">
        <Sparkles size={15} /> Seed demo data
      </Button>
      <Button onClick={onReset} disabled={busy} variant="danger" className="flex-1 justify-center">
        <Trash2 size={15} /> Clear all
      </Button>
    </div>
  );
}
