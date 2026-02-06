"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";

const NAV_TABS = [
  { label: "Overview", href: "/" },
  { label: "Common Queries", href: "/common-queries" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#080a0e]">
      {/* Subtle background */}
      <div className="fixed inset-0 pointer-events-none">
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage:
              "linear-gradient(rgb(74 222 128) 1px, transparent 1px), linear-gradient(90deg, rgb(74 222 128) 1px, transparent 1px)",
            backgroundSize: "72px 72px",
          }}
        />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[500px] bg-green-500 opacity-[0.04] blur-[100px]" />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 py-8 max-w-7xl">
        <header className="mb-10">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-white">
                Sage Analytics
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                100xEngineers · AI Cohort 6
              </p>
            </div>

            <nav className="flex gap-1 bg-white/[0.04] rounded-lg p-1 border border-white/[0.08]">
              {NAV_TABS.map((tab) => {
                const isActive = pathname === tab.href;
                return (
                  <Link
                    key={tab.href}
                    href={tab.href}
                    className={`px-5 py-1.5 rounded-md text-sm font-medium transition-all ${
                      isActive
                        ? "bg-green-500/[0.12] text-green-300"
                        : "text-gray-500 hover:text-gray-300"
                    }`}
                  >
                    {tab.label}
                  </Link>
                );
              })}
            </nav>
          </div>
        </header>

        {children}
      </div>
    </div>
  );
}
