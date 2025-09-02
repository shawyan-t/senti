"use client"

// A tiny wrapper that binds react-plotly to the pre-minified Plotly build.
// Using the dist-min bundle reduces the work Next's minifier must do
// during build while preserving full functionality (including 3D traces).
import createPlotlyComponent from 'react-plotly.js/factory'
// Use the pre-minified distribution shipped with plotly.js to reduce build work
// and keep full functionality (including 3D traces).
// This avoids adding a new dependency like "plotly.js-dist-min".
import Plotly from 'plotly.js/dist/plotly.min.js'

const Plot = createPlotlyComponent(Plotly)
export default Plot
