import * as React from "react";
import { cn } from "./utils";

type DropdownMenuProps = React.HTMLAttributes<HTMLDivElement>;

export function DropdownMenu({ className, ...props }: DropdownMenuProps) {
  return (
    <div
      className={cn("relative inline-block group", className)}
      {...props}
    />
  );
}

export function DropdownMenuTrigger({
  children,
  className,
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("inline-flex", className)}>{children}</div>;
}

export function DropdownMenuContent({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "absolute right-0 z-50 mt-2 hidden min-w-[180px] rounded-xl border border-white/10 bg-zinc-950 p-2 text-sm text-white shadow-xl group-hover:block",
        className
      )}
      {...props}
    />
  );
}

export function DropdownMenuItem({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "cursor-pointer rounded-lg px-3 py-2 text-sm hover:bg-white/10",
        className
      )}
      {...props}
    />
  );
}

export function DropdownMenuLabel({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("px-3 py-2 text-xs text-white/60", className)}
      {...props}
    />
  );
}

export function DropdownMenuSeparator({ className }: { className?: string }) {
  return <div className={cn("my-1 h-px bg-white/10", className)} />;
}
