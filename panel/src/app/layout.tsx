import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Trafo v3 · Filo İzleme (Simülasyon)",
  description: "Yağ sıcaklığı tahmini, açıklamalı alarm ve model sağlığı. Açık veri (ETT) üzerinde simülasyon.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="tr" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
