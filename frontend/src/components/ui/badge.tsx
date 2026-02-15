import * as React from "react";
import { cn } from "./utils";

type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & {
  variant?: "secondary" | "outline" | "destructive";
};

const variants: Record<NonNullable<BadgeProps["variant"]>, string> = {
  secondary: "bg-white/10 text-white",
  outline: "border border-white/10 text-white",
  destructive: "bg-red-600 text-white",
};

export function Badge({ className, variant = "secondary", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium",
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
