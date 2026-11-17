import React, { useState, useEffect } from 'react';
import { 
  UploadCloud, 
  MapPin, 
  Network, 
  Cpu, 
  Download, 
  Layers, 
  Activity, 
  Settings2,
  FileCode,
  Image as ImageIcon
} from 'lucide-react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mode, setMode] = useState('crop'); // 'crop' or 'full'
  const [roadThresh, setRoadThresh] = useState(0.30);
  const [interThresh, setInterThresh] = useState(0.16);
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('overlay'); // 'overlay', 'heatmap', 'json'
  const [systemStatus, setSystemStatus] = useState(null);

  // Health check on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/health')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(() => setSystemStatus({ status: 'offline' }));
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const runExtraction = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const url = `http://127.0.0.1:8000/extract-roads?mode=${mode}&road_thresh=${roadThresh}&inter_thresh=${interThresh}`;
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setActiveTab('overlay');
    } catch (err) {
      alert(`Extraction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify({
      nodes: result.nodes,
      edges: result.edges,
      num_nodes: result.num_nodes,
      num_edges: result.num_edges,
      dimensions: result.dimensions
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `road_graph_${Date.now()}.json`;
    a.click();
  };

  return (
    <div>
      {/* Top Navigation */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Network size={22} color="#fff" />
          </div>
          <div>
            <span className="brand-title">GeoRoad Intelligence</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="brand-badge">
            {systemStatus?.status === 'online' ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }}></span>
                API Connected ({systemStatus.device.toUpperCase()})
              </span>
            ) : (
              <span style={{ color: '#ef4444' }}>API Connecting...</span>
            )}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            Model: <strong style={{ color: '#38bdf8' }}>best_road_model.pth</strong>
          </div>
        </div>
      </nav>

      <div className="app-container">
        {/* Left Control Panel */}
        <div className="controls-card">
          <div className="card-heading">
            <UploadCloud size={16} /> Input Satellite Tile
          </div>

          <div 
            className="dropzone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <input 
              id="file-input" 
              type="file" 
              accept="image/*" 
              style={{ display: 'none' }} 
              onChange={handleFileChange} 
            />
            {previewUrl ? (
              <img 
                src={previewUrl} 
                alt="Preview" 
                style={{ width: '100%', height: '140px', objectFit: 'cover', borderRadius: '8px' }} 
              />
            ) : (
              <>
                <UploadCloud size={36} className="upload-icon" />
                <div className="dropzone-text">Click or Drag Satellite Image</div>
                <div className="dropzone-hint">PNG, JPG, TIFF up to 2048x2048</div>
              </>
            )}
          </div>

          <div className="card-heading">
            <Settings2 size={16} /> Extraction Parameters
          </div>

          {/* Mode Selector */}
          <div className="control-group">
            <div className="control-label-row">
              <span>Execution Canvas</span>
            </div>
            <div className="mode-selector">
              <button 
                className={`mode-btn ${mode === 'crop' ? 'active' : ''}`}
                onClick={() => setMode('crop')}
              >
                512x512 Crop
              </button>
              <button 
                className={`mode-btn ${mode === 'full' ? 'active' : ''}`}
                onClick={() => setMode('full')}
              >
                Full Canvas
              </button>
            </div>
          </div>

          {/* Road Probability Slider */}
          <div className="control-group">
            <div className="control-label-row">
              <span>Road Sensitivity Cutoff</span>
              <span className="control-val">{roadThresh.toFixed(2)}</span>
            </div>
            <input 
              type="range" 
              min="0.10" 
              max="0.60" 
              step="0.02" 
              value={roadThresh} 
              onChange={(e) => setRoadThresh(parseFloat(e.target.value))} 
            />
          </div>

          {/* Intersection Sensitivity Slider */}
          <div className="control-group">
            <div className="control-label-row">
              <span>Intersection Junction Cutoff</span>
              <span className="control-val">{interThresh.toFixed(2)}</span>
            </div>
            <input 
              type="range" 
              min="0.05" 
              max="0.40" 
              step="0.01" 
              value={interThresh} 
              onChange={(e) => setInterThresh(parseFloat(e.target.value))} 
            />
          </div>

          {/* Run Button */}
          <button 
            className="btn-run"
            disabled={!selectedFile || loading}
            onClick={runExtraction}
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Extracting Graph...
              </>
            ) : (
              <>
                <Cpu size={18} />
                Run GeoRoad Analysis
              </>
            )}
          </button>
        </div>

        {/* Right Main Results View */}
        <div className="main-display">
          {/* Top Metric Cards */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Detected Vertices (Nodes)</div>
              <div className="stat-value cyan">{result ? result.num_nodes : '--'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Vector Road Edges</div>
              <div className="stat-value green">{result ? result.num_edges : '--'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Processed Resolution</div>
              <div className="stat-value">
                {result ? `${result.dimensions.width}x${result.dimensions.height}` : '--'}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Avg Degree / Connectivity</div>
              <div className="stat-value">
                {result && result.num_nodes > 0 
                  ? ((result.num_edges * 2) / result.num_nodes).toFixed(2) 
                  : '--'}
              </div>
            </div>
          </div>

          {/* Main Visualizer Window */}
          <div className="canvas-card">
            <div className="canvas-header">
              <div className="view-tabs">
                <button 
                  className={`tab-btn ${activeTab === 'overlay' ? 'active' : ''}`}
                  onClick={() => setActiveTab('overlay')}
                >
                  <Layers size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: -2 }} />
                  Vector Graph Overlay
                </button>
                <button 
                  className={`tab-btn ${activeTab === 'heatmap' ? 'active' : ''}`}
                  onClick={() => setActiveTab('heatmap')}
                >
                  <Activity size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: -2 }} />
                  Probability Heatmap
                </button>
                <button 
                  className={`tab-btn ${activeTab === 'json' ? 'active' : ''}`}
                  onClick={() => setActiveTab('json')}
                >
                  <FileCode size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: -2 }} />
                  Graph JSON ({result ? result.num_nodes : 0} nodes)
                </button>
              </div>

              {result && (
                <button 
                  className="tab-btn"
                  onClick={downloadJson}
                  style={{ background: 'rgba(16, 185, 129, 0.15)', borderColor: '#10b981', color: '#34d399' }}
                >
                  <Download size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: -2 }} />
                  Export JSON Graph
                </button>
              )}
            </div>

            <div className="canvas-body">
              {loading ? (
                <div className="placeholder-box">
                  <div className="spinner" style={{ width: 36, height: 36 }}></div>
                  <div style={{ color: 'var(--text-muted)' }}>Running SAM-Road ViT-B & Topology Decoder on GPU...</div>
                </div>
              ) : result ? (
                <>
                  {activeTab === 'overlay' && (
                    <img 
                      src={result.overlay_base64} 
                      alt="Vector Overlay" 
                      className="result-image" 
                    />
                  )}
                  {activeTab === 'heatmap' && (
                    <img 
                      src={result.heatmap_base64} 
                      alt="Heatmap" 
                      className="result-image" 
                    />
                  )}
                  {activeTab === 'json' && (
                    <pre className="json-viewer">
                      {JSON.stringify({
                        dimensions: result.dimensions,
                        num_nodes: result.num_nodes,
                        num_edges: result.num_edges,
                        nodes: result.nodes.slice(0, 50),
                        edges: result.edges.slice(0, 50),
                        note: result.nodes.length > 50 ? "... truncated for UI preview. Use 'Export JSON Graph' for full file." : undefined
                      }, null, 2)}
                    </pre>
                  )}
                </>
              ) : (
                <div className="placeholder-box">
                  <ImageIcon size={48} />
                  <div>Select or drop a satellite image to extract vector road networks</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

