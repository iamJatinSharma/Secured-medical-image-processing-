import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Authentication - SecureMed AI",
  description: "Login or register to access SecureMed AI",
};

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-teal-50/30 to-slate-100 dark:from-slate-950 dark:via-teal-950/20 dark:to-slate-900">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
      
      {/* Content */}
      <div className="relative z-10 w-full max-w-md px-4">
        {children}
      </div>
    </div>
  );
}
