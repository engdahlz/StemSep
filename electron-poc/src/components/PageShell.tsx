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
        "absolute inset-0 z-20 flex h-full flex-col text-[#fafafa] selection:bg-white/20",
        className,
      )}
    >
      <div className="absolute inset-0 bg-black/40 backdrop-blur-xl" />
      <div
        className={cn(
          "relative z-10 min-h-0 flex-1 scroll-smooth",
          scroll && "overflow-y-auto",
          contentClassName,
        )}
      >
        {children}
      </div>
    </div>
  );
}
