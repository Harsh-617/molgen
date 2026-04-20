"use client";

import { useState } from "react";

const API = "http://localhost:8000";

const EXAMPLES = [
  { name: "Ethanol",   smiles: "CCO",                           desc: "Simple alcohol" },
  { name: "Benzene",   smiles: "c1ccccc1",                      desc: "Aromatic ring" },
  { name: "Aspirin",   smiles: "CC(=O)Oc1ccccc1C(=O)O",        desc: "Common drug" },
  { name: "Caffeine",  smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", desc: "Stimulant" },
  { name: "Glucose",   smiles: "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", desc: "Sugar" },
  { name: "Dopamine",  smiles: "NCCc1ccc(O)c(O)c1",            desc: "Neurotransmitter" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   desc: "NSAID painkiller" },
  { name: "Adenine",   smiles: "Nc1ncnc2[nH]cnc12",            desc: "DNA nucleobase" },
];

interface Props {
  smiles: string;
  is_valid: boolean;
  logp: number | null;
  qed: number | null;
  sa_score: number | null;
  homo_lumo_gap: number | null;
  svg: string | null;
}

function PropBar({ label, value, min, max, color }: {
  label: string; value: number | null; min: number; max: number; color: string;
}) {
  const pct = value != null ? Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100)) : 0;
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
        <span style={{ fontSize: '11px', color: 'var(--muted)' }}>{label}</span>
        <span style={{ fontFamily: 'IBM Plex Mono', fontSize: '11px', color: 'var(--text)' }}>
          {value != null ? value.toFixed(3) : '—'}
        </span>
      </div>
      <div style={{ height: '3px', background: 'var(--border)', borderRadius: '2px' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: '2px', transition: 'width 0.4s ease' }} />
      </div>
    </div>
  );
}

export default function ExplorePage() {
  const [selected, setSelected]   = useState<string | null>(null);
  const [custom, setCustom]       = useState("");
  const [result, setResult]       = useState<Props | null>(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const lookup = async (smiles: string) => {
    setLoading(true);
    setError(null);
    setSelected(smiles);
    try {
      const res = await fetch(`${API}/predict_properties`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '3rem 2rem', maxWidth: '1100px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '24px', fontWeight: 300, letterSpacing: '-0.02em', marginBottom: '0.5rem' }}>
        Molecule Explorer
      </h1>
      <p style={{ color: 'var(--muted)', fontSize: '14px', marginBottom: '2.5rem' }}>
        Browse known molecules and predict their properties using the trained GNN encoder.
      </p>

      {/* Example molecules grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        {EXAMPLES.map(({ name, smiles, desc }) => (
          <button key={smiles} onClick={() => lookup(smiles)}
            style={{
              padding: '1rem', textAlign: 'left',
              background: selected === smiles ? 'rgba(0,229,160,0.08)' : 'var(--surface)',
              border: `1px solid ${selected === smiles ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: '8px', cursor: 'pointer',
              transition: 'all 0.15s',
            }}>
            <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text)', marginBottom: '4px' }}>{name}</p>
            <p style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '8px' }}>{desc}</p>
            <p style={{ fontFamily: 'IBM Plex Mono', fontSize: '10px', color: 'var(--accent)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {smiles}
            </p>
          </button>
        ))}
      </div>

      {/* Custom input */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2.5rem' }}>
        <input
          value={custom}
          onChange={e => setCustom(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && custom.trim() && lookup(custom.trim())}
          placeholder="Or enter any SMILES string…"
          style={{
            flex: 1, padding: '10px 14px',
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: '6px', color: 'var(--text)',
            fontFamily: 'IBM Plex Mono', fontSize: '12px', outline: 'none',
          }}
        />
        <button onClick={() => custom.trim() && lookup(custom.trim())}
          disabled={!custom.trim() || loading}
          style={{
            padding: '10px 20px',
            background: !custom.trim() || loading ? 'var(--border)' : 'var(--accent)',
            color: !custom.trim() || loading ? 'var(--muted)' : '#000',
            border: 'none', borderRadius: '6px',
            fontSize: '12px', fontWeight: 700, cursor: 'pointer',
            letterSpacing: '0.06em', textTransform: 'uppercase',
          }}>
          Lookup
        </button>
      </div>

      {error && (
        <div style={{ padding: '1rem', background: '#1a0810', border: '1px solid var(--danger)', borderRadius: '6px', marginBottom: '1.5rem', color: 'var(--danger)', fontSize: '13px' }}>
          {error}
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          {/* Structure */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
            <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)' }}>
                2D Structure
              </span>
            </div>
            {result.svg ? (
              <div style={{ background: '#fff', padding: '1.5rem', display: 'flex', justifyContent: 'center' }}
                dangerouslySetInnerHTML={{ __html: result.svg }} />
            ) : (
              <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--danger)', fontSize: '13px' }}>
                Cannot render structure
              </div>
            )}
            <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)' }}>
              <p style={{ fontFamily: 'IBM Plex Mono', fontSize: '11px', color: 'var(--muted)', wordBreak: 'break-all' }}>
                {result.smiles}
              </p>
            </div>
          </div>

          {/* Properties */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px' }}>
            <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)' }}>
                GNN Predictions
              </span>
              <span style={{
                fontSize: '10px', padding: '2px 8px', borderRadius: '20px',
                background: result.is_valid ? 'rgba(0,229,160,0.1)' : 'rgba(255,77,106,0.1)',
                color: result.is_valid ? 'var(--accent)' : 'var(--danger)',
                fontWeight: 600,
              }}>
                {result.is_valid ? 'VALID' : 'INVALID'}
              </span>
            </div>
            <div style={{ padding: '1.5rem' }}>
              <PropBar label="LogP (solubility proxy)"   value={result.logp}          min={-5} max={10}  color="var(--accent)" />
              <PropBar label="QED (drug-likeness)"       value={result.qed}           min={0}  max={1}   color="var(--accent2)" />
              <PropBar label="SA Score (synthesisability)" value={result.sa_score}    min={1}  max={10}  color="#f59e0b" />
              <PropBar label="HOMO-LUMO gap (eV)"        value={result.homo_lumo_gap} min={0}  max={1}   color="#a78bfa" />

              <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--bg)', borderRadius: '6px', border: '1px solid var(--border)' }}>
                <p style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '6px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Drug-likeness Assessment
                </p>
                {[
                  { rule: 'LogP ≤ 5',      pass: result.logp != null && result.logp <= 5 },
                  { rule: 'QED ≥ 0.5',     pass: result.qed != null && result.qed >= 0.5 },
                  { rule: 'SA Score ≤ 4',  pass: result.sa_score != null && result.sa_score <= 4 },
                ].map(({ rule, pass }) => (
                  <div key={rule} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <span style={{ color: pass ? 'var(--accent)' : 'var(--danger)', fontSize: '12px' }}>
                      {pass ? '✓' : '✗'}
                    </span>
                    <span style={{ fontSize: '12px', color: pass ? 'var(--text)' : 'var(--muted)' }}>{rule}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--muted)' }}>
          <p style={{ fontSize: '13px' }}>Predicting properties…</p>
        </div>
      )}
    </div>
  );
}