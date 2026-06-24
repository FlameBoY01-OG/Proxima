import * as React from "react";
import { cn } from "../lib/utils";

/* Minimal UI primitives authored in the shadcn/ui pattern (Tailwind classes,
   forwardRef, `cn` for overrides). Swap for the full shadcn library later via
   `npx shadcn@latest add ...` if desired. */

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-line bg-ink-900/70 backdrop-blur-sm shadow-sm",
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-4 pt-3 pb-2", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("text-xs font-semibold uppercase tracking-wider text-muted", className)}
      {...props}
    />
  );
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-4 pb-4", className)} {...props} />;
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "ghost" | "danger";
};

export function Button({ className, variant = "default", ...props }: ButtonProps) {
  const variants = {
    default:
      "bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25 hover:shadow-glow",
    ghost: "text-muted hover:text-fg hover:bg-ink-700 border border-transparent",
    danger:
      "bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20",
  };
  return (
    <button
      className={cn(
        "inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium",
        "transition-all disabled:opacity-40 disabled:pointer-events-none",
        variants[variant],
        className,
      )}
      {...props}
    />
  );
}

export function Badge({
  className,
  color,
  active,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { color?: string; active?: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs",
        "cursor-pointer select-none transition-colors",
        active
          ? "border-fg/30 bg-ink-600 text-fg"
          : "border-line bg-ink-800 text-muted hover:text-fg",
        className,
      )}
      {...props}
    >
      {color && (
        <span
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: color, opacity: active ? 1 : 0.45 }}
        />
      )}
      {props.children}
    </span>
  );
}
