"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { API_URL } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Call" },
  { href: "/calls", label: "History" },
  { href: "/benchmark", label: "Benchmark" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="nav">
      {LINKS.map((l) => {
        const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
        return (
          <Link key={l.href} href={l.href} aria-current={active ? "page" : undefined}>
            {l.label}
          </Link>
        );
      })}
      <a
        href={`${API_URL}/transparency`}
        title="Every component that touches a call: vendor, model, region, licence"
      >
        Transparency
      </a>
    </nav>
  );
}
