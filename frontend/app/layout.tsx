import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Cost Bench",
  description: "One conversation, two voice pipelines: live cost per minute and latency.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="topbar">
            <span className="brand">Voice Cost Bench</span>
            <Nav />
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
