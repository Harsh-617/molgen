"use client";

import { useState } from "react";

const API = "http://localhost:8000";

interface Molecule {
  smiles: string;
  image_b64: string | null;
  svg: string | null;
  predicted_logp: number | null;
  predicted_qed: number | null;
  predicted_sa: number | null;
  predicted_homo_lumo: number | null;
  valid: boolean;
}

interface GenerateResponse {
  molecules: Molecule[];
  n_requested: number;
  n_valid: number;
  generation_time_s: number;
}

function Slider({
  label, hint, value, min, max, step, onChange,
}: {
  label: string; hint: string; value: number;
  min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '12px', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)' }}>
          {label}
        </span>
        <span style={{ fontFamily: 'IBM Plex Mono', fontSize: '13px', color: 'var(--accent)' }}>
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{
          width: '100%', height: '2px', appearance: 'none',
          background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${((value - min) / (max - min)) * 100}%, var(--border) ${((value - min) / (max - min)) * 100}%, var(--border) 100%)`,
          outline: 'none', border: 'none', cursor: 'pointer',
        }}
      />
      <p style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '4px' }}>{hint}</p>
    </div>
  );
}

function MolCard({ mol, target }: { mol: Molecule; target: Record<string, number> }) {
  if (!mol.valid) {
    return (
      <div style={{
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: '8px', padding: '1rem',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: '240px', color: 'var(--danger)', fontSize: '12px',
      }}>
        Invalid molecule
      </div>
    );
  }

  const props = [
    { key: 'LogP',      pred: mol.predicted_logp,      tgt: target.logp },
    { key: 'QED',       pred: mol.predicted_qed,       tgt: target.qed },
    { key: 'SA Score',  pred: mol.predicted_sa,        tgt: target.sa },
    { key: 'HOMO-LUMO', pred: mol.predicted_homo_lumo, tgt: target.homo_lumo },
  ];

  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: '8px', overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
      onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
    >
      {mol.image_b64 && (
        <div style={{ background: '#fff', padding: '8px', textAlign: 'center' }}>
          <img src={`data:image/png;base64,${mol.image_b64}`} alt="molecule"
            style={{ width: '100%', maxHeight: '160px', objectFit: 'contain' }} />
        </div>
      )}
      <div style={{ padding: '0.75rem' }}>
        <p className="mono" style={{
          fontSize: '10px', color: 'var(--muted)', marginBottom: '0.75rem',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {mol.smiles}
        </p>
        {props.map(({ key, pred, tgt }) => (
          <div key={key} style={{ marginBottom: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
              <span style={{ fontSize: '10px', color: 'var(--muted)' }}>{key}</span>
              <span className="mono" style={{ fontSize: '10px', color: 'var(--text)' }}>
                {pred != null ? pred.toFixed(2) : '—'}
                <span style={{ color: 'var(--muted)' }}> / {tgt.toFixed(2)}</span>
              </span>
            </div>
            {pred != null && (
              <div style={{ height: '2px', background: 'var(--border)', borderRadius: '1px' }}>
                <div style={{
                  height: '100%', borderRadius: '1px',
                  background: Math.abs(pred - tgt) < 0.5 ? 'var(--accent)' : 'var(--accent2)',
                  width: `${Math.min(100, (1 - Math.min(1, Math.abs(pred - tgt))) * 100)}%`,
                }} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function GeneratePage() {
  const [logp, setLogp]         = useState(2.0);
  const [qed, setQed]           = useState(0.7);
  const [sa, setSa]             = useState(3.0);
  const [homoLumo, setHomoLumo] = useState(0.27);
  const [nMols, setNMols]       = useState(8);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState<GenerateResponse | null>(null);
  const [error, setError]       = useState<string | null>(null);

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_logp: logp,
          target_qed: qed,
          target_sa: sa,
          target_homo_lumo: homoLumo,
          n_molecules: nMols,
        }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const target = { logp, qed, sa, homo_lumo: homoLumo };

  return (
    <div style={{ display: 'flex', minHeight: 'calc(100vh - 56px)' }}>

      {/* Sidebar */}
      <aside style={{
        width: '300px', flexShrink: 0,
        borderRight: '1px solid var(--border)',
        padding: '2rem 1.5rem',
        background: 'var(--surface)',
      }}>
        <h2 style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: '2rem' }}>
          Target Properties
        </h2>

        <Slider label="LogP" hint="Water solubility proxy (–5 to 10)"
          value={logp} min={-5} max={10} step={0.1} onChange={setLogp} />
        <Slider label="QED" hint="Drug-likeness score (0 to 1)"
          value={qed} min={0} max={1} step={0.01} onChange={setQed} />
        <Slider label="SA Score" hint="Synthetic accessibility (1=easy, 10=hard)"
          value={sa} min={1} max={10} step={0.1} onChange={setSa} />
        <Slider label="HOMO-LUMO Gap" hint="Electronic property in eV"
          value={homoLumo} min={0} max={1} step={0.01} onChange={setHomoLumo} />

        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)' }}>
              Count
            </span>
            <span className="mono" style={{ fontSize: '13px', color: 'var(--accent)' }}>{nMols}</span>
          </div>
          <input type="range" min={1} max={20} step={1} value={nMols}
            onChange={e => setNMols(parseInt(e.target.value))}
            style={{ width: '100%', height: '2px', appearance: 'none', outline: 'none', border: 'none', cursor: 'pointer',
              background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${((nMols - 1) / 19) * 100}%, var(--border) ${((nMols - 1) / 19) * 100}%, var(--border) 100%)`,
            }} />
        </div>

        <button onClick={generate} disabled={loading}
          style={{
            width: '100%', padding: '12px',
            background: loading ? 'var(--border)' : 'var(--accent)',
            color: loading ? 'var(--muted)' : '#000',
            border: 'none', borderRadius: '6px',
            fontSize: '13px', fontWeight: 700, letterSpacing: '0.08em',
            textTransform: 'uppercase', cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
          }}>
          {loading ? 'Generating…' : 'Generate'}
        </button>

        {result && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--bg)', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <p className="mono" style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '4px' }}>
              Valid: <span style={{ color: 'var(--accent)' }}>{result.n_valid}/{result.n_requested}</span>
            </p>
            <p className="mono" style={{ fontSize: '11px', color: 'var(--muted)' }}>
              Time: <span style={{ color: 'var(--text)' }}>{result.generation_time_s}s</span>
            </p>
          </div>
        )}
      </aside>

      {/* Main */}
      <main style={{ flex: 1, padding: '2rem', overflow: 'auto' }}>
        <div style={{ maxWidth: '1200px' }}>
          <h1 style={{ fontSize: '24px', fontWeight: 300, marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>
            Molecule Generator
          </h1>
          <p style={{ color: 'var(--muted)', fontSize: '14px', marginBottom: '2rem' }}>
            Set target properties and generate novel drug-like molecules using graph diffusion.
          </p>

          {error && (
            <div style={{ padding: '1rem', background: '#1a0810', border: '1px solid var(--danger)', borderRadius: '6px', marginBottom: '1.5rem', color: 'var(--danger)', fontSize: '13px' }}>
              {error} — make sure the API is running on port 8000.
            </div>
          )}

          {!result && !loading && (
            <div style={{
              border: '1px dashed var(--border)', borderRadius: '8px',
              padding: '4rem', textAlign: 'center', color: 'var(--muted)',
            }}>
              <p style={{ fontSize: '48px', marginBottom: '1rem' }}>⬡</p>
              <p style={{ fontSize: '14px' }}>Set your target properties and click Generate</p>
            </div>
          )}

          {loading && (
            <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--muted)' }}>
              <p style={{ fontSize: '48px', marginBottom: '1rem', animation: 'spin 2s linear infinite' }}>◈</p>
              <p style={{ fontSize: '14px' }}>Running diffusion sampling…</p>
              <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
            </div>
          )}

          {result && !loading && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: '1rem',
            }}>
              {result.molecules.map((mol, i) => (
                <MolCard key={i} mol={mol} target={target} />
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}