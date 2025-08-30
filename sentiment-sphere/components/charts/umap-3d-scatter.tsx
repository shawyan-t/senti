// @ts-expect-error: No types for react-plotly.js
import Plot from 'react-plotly.js';
import React from 'react';
import { UMAP } from 'umap-js';

interface UMAP3DScatterProps {
  embeddings: number[][]; // shape: [n_samples, n_features]
  labels?: string[];
  colors?: string[];
}

export const UMAP3DScatter: React.FC<UMAP3DScatterProps> = ({ embeddings, labels, colors }) => {
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

  return (
    <Plot
      data={[
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
            opacity: 0.85,
          },
          text: labels || undefined,
        },
      ]}
      layout={{
        autosize: true,
        height: 500,
        title: '3D UMAP Topic/Sentiment Landscape',
        scene: {
          xaxis: { title: 'UMAP-1' },
          yaxis: { title: 'UMAP-2' },
          zaxis: { title: 'UMAP-3' },
        },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        paper_bgcolor: 'rgba(0,0,0,0.85)',
        plot_bgcolor: 'rgba(0,0,0,0.85)',
        font: { color: '#a7f3d0' },
      }}
      config={{ responsive: true }}
      style={{ width: '100%', minHeight: 500 }}
    />
  );
}; 