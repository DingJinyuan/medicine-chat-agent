import { ShieldAlert } from "lucide-react";

export function DisclaimerFooter() {
  return (
    <div className="flex items-center justify-center gap-1.5 border-t bg-muted/30 px-4 py-2 text-center text-[11px] text-muted-foreground">
      <ShieldAlert className="h-3 w-3 shrink-0" />
      <p>
        Portfolio demo, not medical advice or a real clinical tool. For emergencies, call your local
        emergency number.
      </p>
    </div>
  );
}
