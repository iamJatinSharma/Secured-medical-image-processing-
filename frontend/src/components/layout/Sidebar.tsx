"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Home,
  Image,
  Stethoscope,
  Shield,
  Brain,
  BarChart3,
  LineChart,
  Activity,
  FileText,
  Users,
  Scissors,
  ChevronLeft,
  ChevronRight,
  HeartPulse,
} from "lucide-react";

const navItems = [
  { href: "/dashboard", label: "Home", icon: Home },
  { href: "/processing", label: "Image Processing", icon: Image },
  { href: "/detection", label: "Disease Detection", icon: Stethoscope },
  { href: "/security", label: "Security", icon: Shield },
  { href: "/xai", label: "Explainable AI", icon: Brain },
  { href: "/evaluation", label: "Evaluation Metrics", icon: BarChart3 },
  { href: "/visualizations", label: "Visualizations", icon: LineChart },
  { href: "/training", label: "Training Monitor", icon: Activity },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/users", label: "User Management", icon: Users },
  { href: "/segmentation", label: "Segmentation", icon: Scissors },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 72 : 256 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="fixed left-0 top-0 z-40 h-screen bg-sidebar flex flex-col border-r border-sidebar-border"
    >
      {/* Logo Section */}
      <div className="flex items-center h-16 px-4 border-b border-sidebar-border">
        <motion.div
          initial={false}
          animate={{ opacity: 1 }}
          className="flex items-center gap-3"
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/20">
            <HeartPulse className="w-6 h-6 text-primary" />
          </div>
          <AnimatePresence mode="wait">
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="flex flex-col"
              >
                <span className="font-bold text-lg text-sidebar-foreground tracking-tight">
                  SecureMed
                </span>
                <span className="text-xs text-sidebar-foreground/60 -mt-0.5">
                  AI Platform
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4 px-2">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;

            return (
              <li key={item.href}>
                <Link href={item.href}>
                  <motion.div
                    className={cn(
                      "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors relative",
                      isActive
                        ? "bg-sidebar-primary text-sidebar-primary-foreground"
                        : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                    )}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="activeNav"
                        className="absolute inset-0 bg-sidebar-primary rounded-lg"
                        transition={{ type: "spring", stiffness: 380, damping: 30 }}
                      />
                    )}
                    <Icon className="w-5 h-5 relative z-10 flex-shrink-0" />
                    <AnimatePresence mode="wait">
                      {!collapsed && (
                        <motion.span
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          transition={{ duration: 0.2 }}
                          className="relative z-10 text-sm font-medium whitespace-nowrap"
                        >
                          {item.label}
                        </motion.span>
                      )}
                    </AnimatePresence>
                  </motion.div>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Collapse Toggle */}
      <div className="p-2 border-t border-sidebar-border">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5" />
              <span className="text-sm font-medium">Collapse</span>
            </>
          )}
        </button>
      </div>
    </motion.aside>
  );
}
