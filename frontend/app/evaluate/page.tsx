"use client";

import { useState } from "react";

const API = "http://localhost:8000";

interface PredictResponse {
  smiles: string;
  is_valid: boolean;
  logp: number | null;
  qed: number | null;
  sa_score: number | null;
  homo_lumo_gap: number | null;
  svg: string | null;
}

export default function EvaluatePage() {
  const [smiles, setSmiles]     = useState("");
  const [result, setResult]     = useState<PredictResponse | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);

  const predict = async () => {
    if (!smiles.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/predict_properties`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: smiles.trim() }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const props = result ? [
    { label: 'LogP',            value: result.logp,          hint: 'Water solubility proxy' },
    { label: 'QED',             value: result.qed,           hint: 'Drug-likeness (0–1)' },
    { label: 'SA Score',        value: result.sa_score,      hint: 'Synthetic accessibility (1–10)' },
    { label: 'HOMO-LUMO Gap',   value: result.homo_lumo_gap, hint: 'Electronic property (eV)' },
  ] : [];

  return (
    <div style={{ padding: '3rem 2rem', maxWidth: '900px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '24px', fontWeight: 300, letterSpacing: '-0.02em', marginBottom: '0.5rem' }}>
        Property Predictor
      </h1>
      <p style={{ color: 'var(--muted)', fontSize: '14px', marginBottom: '2.5rem' }}>
        Enter a SMILES string to predict molecular properties using the trained GNN encoder.
      </p>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
        <input
          value={smiles}
          onChange={e => setSmiles(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && predict()}
          placeholder="Enter SMILES string, e.g. CCO"
          className="mono"
          style={{
            flex: 1, padding: '12px 16px',
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: '6px', color: 'var(--text)', fontSize: '13px',
            outline: 'none',
          }}
        />
        <button onClick={predict} disabled={loading || !smiles.trim()}
          style={{
            padding: '12px 24px',
            background: loading || !smiles.trim() ? 'var(--border)' : 'var(--accent)',
            color: loading || !smiles.trim() ? 'var(--muted)' : '#000',
            border: 'none', borderRadius: '6px',
            fontSize: '13px', fontWeight: 700, letterSpacing: '0.06em',
            textTransform: 'uppercase', cursor: 'pointer',
          }}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </div>

      {/* Quick examples */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '2rem', flexWrap: 'wrap' }}>
        {['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'].map(s => (
          <button key={s} onClick={() => setSmiles(s)}
            className="mono"
            style={{
              padding: '4px 10px', background: 'var(--surface)',
              border: '1px solid var(--border)', borderRadius: '4px',
              color: 'var(--muted)', fontSize: '11px', cursor: 'pointer',
            }}>
            {s}
          </button>
        ))}
      </div>

      {error && (
        <div style={{ padding: '1rem', background: '#1a0810', border: '1px solid var(--danger)', borderRadius: '6px', marginBottom: '1.5rem', color: 'var(--danger)', fontSize: '13px' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>

          {/* Structure */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)' }}>
                Structure
              </span>
            </div>
            {result.svg ? (
              <div style={{ background: '#fff', padding: '1rem' }}
                dangerouslySetInnerHTML={{ __html: result.svg }} />
            ) : (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--danger)', fontSize: '13px' }}>
                Invalid SMILES
              </div>
            )}
            <div style={{ padding: '12px 16px' }}>
              <p className="mono" style={{ fontSize: '11px', color: 'var(--muted)', wordBreak: 'break-all' }}>
                {result.smiles}
              </p>
            </div>
          </div>

          {/* Properties */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)' }}>
                Predicted Properties
              </span>
            </div>
            <div style={{ padding: '1rem' }}>
              {result.is_valid ? props.map(({ label, value, hint }) => (
                <div key={label} style={{ padding: '1rem 0', borderBottom: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text)' }}>{label}</span>
                    <span className="mono" style={{ fontSize: '14px', color: 'var(--accent)' }}>
                      {value != null ? value.toFixed(3) : '—'}
                    </span>
                  </div>
                  <p style={{ fontSize: '11px', color: 'var(--muted)' }}>{hint}</p>
                </div>
              )) : (
                <p style={{ padding: '1rem', color: 'var(--danger)', fontSize: '13px' }}>
                  Invalid SMILES string. Check the structure and try again.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}