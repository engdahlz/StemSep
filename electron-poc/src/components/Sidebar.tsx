import { useEffect, useState } from "react";
import {
  ChevronRight,
  Clock,
  Headphones,
  Home,
  Library,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
} from "lucide-react";

import type { Page } from "../types/navigation";

interface SidebarProps {
  currentPage: Page;
  onPageChange: (page: Page) => void;
}

type NavItem = {
  id: Page;
  label: string;
  icon: typeof Home;
};

const navItems: NavItem[] = [
  { id: "home", label: "Home", icon: Home },
  { id: "models", label: "Model Library", icon: Library },
  { id: "history", label: "History", icon: Clock },
  { id: "results", label: "Result Studio", icon: Headphones },
];

export function Sidebar({ currentPage, onPageChange }: SidebarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [shouldRender, setShouldRender] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setShouldRender(true);
      const frame = window.requestAnimationFrame(() => {
        setIsVisible(true);
      });
      return () => window.cancelAnimationFrame(frame);
    }

    setIsVisible(false);
    const timer = window.setTimeout(() => {
      setShouldRender(false);
    }, 280);

    return () => window.clearTimeout(timer);
  }, [isOpen]);

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen((value) => !value)}
        className={`fixed z-50 rounded-[14px] border p-2.5 shadow-lg transition-all duration-300 ${
          isVisible ? "left-7 top-7" : "left-5 top-5"
        } ${
          isOpen || shouldRender
            ? "border-white/50 bg-white/80 text-slate-700 shadow-black/10 backdrop-blur-xl"
            : "border-white/30 bg-white/20 text-white shadow-black/10 backdrop-blur-md hover:border-white/50 hover:bg-white/30"
        }`}
        aria-label={isOpen ? "Close navigation" : "Open navigation"}
      >
        {isOpen ? (
          <PanelLeftClose className="h-5 w-5" />
        ) : (
          <PanelLeftOpen className="h-5 w-5" />
        )}
      </button>

      {shouldRender && (
        <>
          <button
            type="button"
            className={`stemsep-sidebar-backdrop fixed inset-0 z-30 transition-opacity duration-300 ${
              isVisible ? "opacity-100" : "opacity-0"
            }`}
            onClick={() => {
              setIsOpen(false);
            }}
            aria-label="Close navigation overlay"
          />
          <aside
            className={`stemsep-sidebar-shell fixed bottom-4 left-4 top-4 z-40 flex w-[372px] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-[2rem] border border-white/55 bg-[rgba(250,248,252,0.82)] text-slate-900 shadow-[0_40px_120px_rgba(0,0,0,0.18)] backdrop-blur-2xl transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] ${
              isVisible
                ? "translate-x-0 opacity-100"
                : "-translate-x-[calc(100%+1.5rem)] opacity-0"
            }`}
          >
            <div className="px-6 pb-5 pt-20">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="rounded-full border border-slate-300/70 bg-white/65 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                  Navigation
                </span>
                <span className="rounded-full border border-slate-300/70 bg-white/55 px-3 py-1 text-[11px] text-slate-500">
                  Studio Shell
                </span>
              </div>
              <h2 className="text-[24px] font-normal tracking-[-0.8px] text-slate-900">
                StemSep
              </h2>
              <p className="mt-2 max-w-[260px] text-[13px] leading-[1.45] text-slate-500">
                Move between separation, models, history and results without
                leaving the current visual context.
              </p>
            </div>

            <nav className="flex flex-1 flex-col gap-2 px-4">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPage === item.id;

                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      onPageChange(item.id);
                      setIsOpen(false);
                    }}
                    className={`stemsep-sidebar-item flex items-center gap-3 rounded-[1.2rem] px-4 py-3 text-left transition-all duration-200 ${
                      isActive
                        ? "border border-white/80 bg-white/88 text-slate-900 shadow-[0_18px_40px_rgba(0,0,0,0.10)]"
                        : "border border-transparent text-slate-500 hover:border-white/80 hover:bg-white/54 hover:text-slate-800"
                    }`}
                    style={{
                      transitionDelay: isVisible
                        ? `${navItems.indexOf(item) * 28}ms`
                        : "0ms",
                    }}
                  >
                    <div
                      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[1rem] ${
                        isActive ? "bg-slate-900/[0.06]" : "bg-white/70"
                      }`}
                    >
                      <Icon className="h-[18px] w-[18px]" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-[14px] tracking-[-0.2px]">
                        {item.label}
                      </div>
                      <div className="mt-0.5 text-[11px] text-slate-500">
                        {item.id === "home"
                          ? "Start new separation"
                          : item.id === "models"
                            ? "Browse installed and available models"
                            : item.id === "history"
                              ? "Inspect previous runs"
                              : "Preview and export stems"}
                      </div>
                    </div>
                    <ChevronRight
                      className={`h-4 w-4 transition-transform ${
                        isActive ? "translate-x-0 text-slate-500" : "-translate-x-1 text-slate-300"
                      }`}
                    />
                  </button>
                );
              })}
            </nav>

            <div className="px-4 pb-4 pt-3">
              <button
                type="button"
                onClick={() => {
                  onPageChange("settings");
                  setIsOpen(false);
                }}
                className={`flex w-full items-center gap-3 rounded-[1.2rem] border px-4 py-3 text-left transition-all duration-200 ${
                  currentPage === "settings"
                    ? "border-white/80 bg-white/88 text-slate-900 shadow-[0_18px_40px_rgba(0,0,0,0.10)]"
                    : "border-white/70 bg-white/45 text-slate-600 hover:bg-white/60 hover:text-slate-900"
                }`}
              >
                <div
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[1rem] ${
                    currentPage === "settings" ? "bg-slate-900/[0.06]" : "bg-white/70"
                  }`}
                >
                  <Settings className="h-[18px] w-[18px]" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[14px] tracking-[-0.2px]">Settings</div>
                  <div className="mt-0.5 text-[11px] text-slate-500">
                    Machine defaults, output and app behavior
                  </div>
                </div>
                <ChevronRight
                  className={`h-4 w-4 ${
                    currentPage === "settings" ? "text-slate-500" : "text-slate-300"
                  }`}
                />
              </button>
            </div>

            <div className="mx-4 mb-4 rounded-[1.2rem] border border-white/70 bg-white/45 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.16em] text-slate-400">
                Workspace
              </div>
              <div className="mt-1 text-[13px] text-slate-600">
                Separation, model management and result review in one shell.
              </div>
            </div>
          </aside>
        </>
      )}
    </>
  );
}
