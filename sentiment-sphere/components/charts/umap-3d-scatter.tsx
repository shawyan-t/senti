import dynamic from 'next/dynamic'
import React from 'react';
import { UMAP } from 'umap-js';

interface UMAP3DScatterProps {
  embeddings: number[][]; // shape: [n_samples, n_features]
  labels?: string[];
  colors?: string[];
  hovertexts?: string[];
}

const Plot = dynamic(() => import('./plotly-wrapper'), { ssr: false })

export const UMAP3DScatter: React.FC<UMAP3DScatterProps> = ({ embeddings, labels, colors, hovertexts }) => {
  // Only run UMAP if embeddings are present and have enough dimensions
  const [umapResult, setUmapResult] = React.useState<number[][]>([]);

  React.useEffect(() => {
    if (embeddings && embeddings.length > 0 && embeddings[0].length > 3) {
      const umap = new UMAP({ nComponents: 3, nNeighbors: 15, minDist: 0.1 });
      const result = umap.fit(embeddings);
      setUmapResult(result);
    } else if (embeddings && embeddings.length > 0 && embeddings[0].length === 3) {
      setUmapResult(embeddings);
    } else {
      setUmapResult([]);
    }
  }, [embeddings]);

  if (!umapResult || umapResult.length === 0) {
    return <div className="text-gray-400 text-center">No embedding data available for UMAP visualization.</div>;
  }

  const x = umapResult.map((v) => v[0]);
  const y = umapResult.map((v) => v[1]);
  const z = umapResult.map((v) => v[2]);

  // Helper: convert hex color (e.g., #10B981) to rgba with alpha
  const hexToRgba = (hex: string, alpha = 0.35) => {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!m) return `rgba(31,41,55,${alpha})`; // slate-800 fallback
    const r = parseInt(m[1], 16);
    const g = parseInt(m[2], 16);
    const b = parseInt(m[3], 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Build main scatter trace
  const traces: any[] = [
    {
      x,
      y,
      z,
      mode: 'markers',
      type: 'scatter3d',
      marker: {
        size: 6,
        color: colors || undefined,
        colorscale: 'Viridis',
        opacity: 0.9,
      },
      text: (hovertexts && hovertexts.length === x.length) ? hovertexts : (labels || undefined),
      hovertemplate: '%{text}<extra></extra>',
    },
  ];

  // Optional tracers: connect nearest neighbors within same color group if close
  if (colors && colors.length === umapResult.length) {
    // Group indices by color
    const groups = new Map<string, number[]>();
    colors.forEach((c, i) => {
      if (!groups.has(c)) groups.set(c, []);
      groups.get(c)!.push(i);
    });

    groups.forEach((indices, color) => {
      if (indices.length < 2) return;

      // For each point, find nearest neighbor within the same color group
      const mins: number[] = [];
      const nn: number[] = new Array(indices.length).fill(-1);

      for (let a = 0; a < indices.length; a++) {
        const ia = indices[a];
        let best = Infinity;
        let bestIdx = -1;
        const ax = umapResult[ia][0];
        const ay = umapResult[ia][1];
        const az = umapResult[ia][2];
        for (let b = 0; b < indices.length; b++) {
          if (a === b) continue;
          const ib = indices[b];
          const dx = umapResult[ib][0] - ax;
          const dy = umapResult[ib][1] - ay;
          const dz = umapResult[ib][2] - az;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < best) {
            best = d2;
            bestIdx = b;
          }
        }
        mins.push(Math.sqrt(best));
        nn[a] = bestIdx;
      }

      // Use median of nearest distances as a threshold to avoid clutter
      const sorted = [...mins].sort((a, b) => a - b);
      const med = sorted[Math.floor(sorted.length / 2)] || 0;

      const ex: (number | null)[] = [];
      const ey: (number | null)[] = [];
      const ez: (number | null)[] = [];
      for (let a = 0; a < indices.length; a++) {
        const bIdx = nn[a];
        if (bIdx < 0) continue;
        const dist = mins[a];
        // Only connect if within median nearest-neighbor distance
        if (dist <= med) {
          const ia = indices[a];
          const ib = indices[bIdx];
          ex.push(umapResult[ia][0], umapResult[ib][0], null);
          ey.push(umapResult[ia][1], umapResult[ib][1], null);
          ez.push(umapResult[ia][2], umapResult[ib][2], null);
        }
      }

      if (ex.length > 0) {
        traces.push({
          x: ex,
          y: ey,
          z: ez,
          mode: 'lines',
          type: 'scatter3d',
          line: { color: hexToRgba(color, 0.35), width: 2 },
          hoverinfo: 'skip',
          showlegend: false,
        });
      }
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 500,
        title: '3D UMAP Topic/Sentiment Landscape',
        scene: {
          bgcolor: '#ffffff',
          xaxis: { title: 'UMAP-1', backgroundcolor: '#ffffff', gridcolor: '#000000', zerolinecolor: '#000000', color: '#000000' },
          yaxis: { title: 'UMAP-2', backgroundcolor: '#ffffff', gridcolor: '#000000', zerolinecolor: '#000000', color: '#000000' },
          zaxis: { title: 'UMAP-3', backgroundcolor: '#ffffff', gridcolor: '#000000', zerolinecolor: '#000000', color: '#000000' },
        },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#334155' },
      }}
      config={{ responsive: true }}
      style={{ width: '100%', minHeight: 500, background: 'white' }}
    />
  );
};
