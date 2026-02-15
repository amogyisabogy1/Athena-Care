import * as React from "react";
import { cn } from "./utils";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "secondary" | "outline" | "destructive";
  size?: "sm" | "md";
};

const variants: Record<NonNullable<ButtonProps["variant"]>, string> = {
  default: "bg-blue-600 text-white hover:bg-blue-700",
  secondary: "bg-white/10 text-white hover:bg-white/20",
  outline: "border border-white/10 text-white hover:bg-white/5",
  destructive: "bg-red-600 text-white hover:bg-red-700",
};

const sizes: Record<NonNullable<ButtonProps["size"]>, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-10 px-4 text-sm",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center rounded-xl transition focus:outline-none focus:ring-2 focus:ring-white/20",
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    />
  )
);
Button.displayName = "Button";
