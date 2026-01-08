import type { ReactNode } from "react";

import { cn } from "../lib/utils";

type PageShellProps = {
  children: ReactNode;
  className?: string;
  contentClassName?: string;
  scroll?: boolean;
};

export function PageShell({
  children,
  className,
  contentClassName,
  scroll = true,
}: PageShellProps) {
  return (
    <div
      className={cn(
        "h-full flex flex-col bg-background text-foreground selection:bg-primary/30",
        className,
      )}
    >
      <div
        className={cn(
          "flex-1 relative scroll-smooth",
          scroll && "overflow-y-auto",
          contentClassName,
        )}
      >
        {children}
      </div>
    </div>
  );
}
