import { useState } from 'react';
import axios from 'axios';

// Simple inline styles for the MVP (startup speed > perfection)
const styles = {
  container: { display: 'flex', height: '100vh', fontFamily: 'Arial, sans-serif', backgroundColor: '#1a1a1a', color: '#fff' },
  panel: { padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' },
  left: { width: '350px', borderRight: '1px solid #333', backgroundColor: '#252525' },
  right: { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#121212' },
  input: { padding: '12px', borderRadius: '5px', border: '1px solid #444', backgroundColor: '#333', color: '#fff', width: '100%' },
  button: { padding: '15px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: '#fff', cursor: 'pointer', fontWeight: 'bold', fontSize: '16px' },
  imagePreview: { maxWidth: '100%', maxHeight: '80vh', borderRadius: '8px', boxShadow: '0 4px 20px rgba(0,0,0,0.5)' },
  label: { fontSize: '14px', color: '#aaa', marginBottom: '5px', display: 'block' },
  status: { color: '#00ff88', fontSize: '14px', marginTop: '10px' }
};

function App() {
  const [prompt, setPrompt] = useState('');
  const [file, setFile] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('Ready');
  const [batchSize, setBatchSize] = useState(1); // Default to 1 (Single Shot)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleGenerate = async () => {
    if (!prompt || !file) {
      alert("Please provide both a prompt and a reference layout image.");
      return;
    }
  
    setLoading(true);
    setStatus(batchSize > 1 ? `Director: Planning ${batchSize} variations...` : 'Director: Analyzing intent...'); 
  
    const formData = new FormData();
    formData.append('intent', prompt);
    formData.append('control_image', file);
  
    // Determine which endpoint to hit
    const endpoint = batchSize > 1 ? 'http://127.0.0.1:8000/generate_batch' : 'http://127.0.0.1:8000/generate';
  
    if (batchSize > 1) {
        formData.append('batch_size', batchSize);
    }
  
    try {
      setStatus(`Artist: Forging dataset (${batchSize} items)... This may take time.`); 
  
      const response = await axios.post(endpoint, formData, {
        responseType: 'blob' 
      });
  
      // Download Logic (Same as before)
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', batchSize > 1 ? 'edgeforge_dataset.zip' : 'edgeforge_asset.zip');
      document.body.appendChild(link);
      link.click();
  
      setStatus('Success! Dataset downloaded.');
      setGeneratedImage(null); 
  
    } catch (error) {
      console.error("Error:", error);
      setStatus('Error: Generation failed. Check backend console.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      {/* Left Panel: Controls */}
      <div style={{ ...styles.panel, ...styles.left }}>
        <h2>EdgeForge AI ⚒️</h2>
        
        <div>
          <label style={styles.label}>1. Reference Layout (Canny Edge)</label>
          <input type="file" accept="image/*" onChange={handleFileChange} style={styles.input} />
        </div>

        <div>
          <label style={styles.label}>2. Director's Intent</label>
          <textarea 
            rows="4"
            placeholder="e.g. 'A futuristic car, hard to see, heavy fog'" 
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            style={{...styles.input, resize: 'none'}}
          />
        </div>
        <div>
        <label style={styles.label}>3. Batch Size: {batchSize} images</label>
        <input 
          type="range" 
          min="1" 
          max="10" 
          value={batchSize} 
          onChange={(e) => setBatchSize(e.target.value)}
          style={{ width: '100%', cursor: 'pointer' }}
        />
        </div>
        <button 
          onClick={handleGenerate} 
          disabled={loading}
          style={{ ...styles.button, opacity: loading ? 0.7 : 1 }}
        >
          {loading ? 'Forging...' : 'Generate Asset'}
        </button>

        <div style={styles.status}>Status: {status}</div>
      </div>

      {/* Right Panel: Viewport */}
      <div style={{ ...styles.panel, ...styles.right }}>
        {generatedImage ? (
          <img src={generatedImage} alt="Generated Asset" style={styles.imagePreview} />
        ) : (
          <div style={{ color: '#444' }}>Upload layout and prompt to begin.</div>
        )}
      </div>
    </div>
  );
}

export default App;