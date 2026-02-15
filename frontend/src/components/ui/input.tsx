import * as React from "react";
import { cn } from "./utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-10 w-full rounded-xl border border-white/10 bg-white/5 px-3 text-sm text-white placeholder:text-white/40",
      className
    )}
    {...props}
  />
));
Input.displayName = "Input";
