import * as React from "react";
import { cn } from "./utils";

type TabsContextValue = {
  value: string;
  setValue: (value: string) => void;
};

const TabsContext = React.createContext<TabsContextValue | null>(null);

type TabsProps = {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
} & React.HTMLAttributes<HTMLDivElement>;

export function Tabs({ value, onValueChange, children, className }: TabsProps) {
  return (
    <TabsContext.Provider value={{ value, setValue: onValueChange }}>
      <div className={cn("inline-flex", className)}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("inline-flex items-center gap-1 rounded-xl bg-white/5 p-1", className)}
      {...props}
    />
  );
}

export function TabsTrigger({
  value,
  className,
  children,
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { value: string }) {
  const ctx = React.useContext(TabsContext);
  const active = ctx?.value === value;
  return (
    <button
      type="button"
      onClick={() => ctx?.setValue(value)}
      className={cn(
        "rounded-lg px-3 py-1.5 text-xs font-medium transition",
        active ? "bg-white/10 text-white" : "text-white/60 hover:text-white",
        className
      )}
    >
      {children}
    </button>
  );
}
