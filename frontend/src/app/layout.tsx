import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "SecureMed AI - Medical Image Processing Platform",
  description: "Secure, optimized, and intelligent platform for medical image processing with AI-powered disease detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased">
        <TooltipProvider>
          {children}
        </TooltipProvider>
      </body>
    </html>
  );
}
