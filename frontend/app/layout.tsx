import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MolGen — Molecular Generation",
  description: "Conditional molecular generation with graph diffusion networks",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <nav style={{
          borderBottom: '1px solid var(--border)',
          padding: '0 2rem',
          height: '56px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--surface)',
          position: 'sticky',
          top: 0,
          zIndex: 100,
        }}>
          <span style={{ fontFamily: 'IBM Plex Mono', fontSize: '14px', color: 'var(--accent)', letterSpacing: '0.05em' }}>
            ◈ MOLGEN
          </span>
          <div style={{ display: 'flex', gap: '2rem' }}>
            {[
              { href: '/', label: 'Generate' },
              { href: '/evaluate', label: 'Evaluate' },
              { href: '/explore', label: 'Explore' },
            ].map(({ href, label }) => (
              <a key={href} href={href} style={{
                color: 'var(--muted)',
                textDecoration: 'none',
                fontSize: '13px',
                fontWeight: 500,
                letterSpacing: '0.05em',
                textTransform: 'uppercase',
              }}>
                {label}
              </a>
            ))}
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}