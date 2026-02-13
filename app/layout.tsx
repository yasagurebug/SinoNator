import type { Metadata } from "next";
import { Baloo_2, Zen_Kaku_Gothic_New } from "next/font/google";
import "./globals.css";

const titleFont = Baloo_2({
  subsets: ["latin-ext"],
  variable: "--font-title",
  weight: ["700"]
});

const bodyFont = Zen_Kaku_Gothic_New({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "700"]
});

export const metadata: Metadata = {
  title: "シノネーター",
  description: "しののめにこアーカイブをアキネーター風に探す"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body className={`${titleFont.variable} ${bodyFont.variable}`}>{children}</body>
    </html>
  );
}
