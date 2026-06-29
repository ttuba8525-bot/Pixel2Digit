/* ==========================================================================
   NEUROVISION XAI LAB - JAVASCRIPT SYSTEM
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Systems
    initParticleBackground();
    initHeroAnn();
    initDrawingCanvas();
    initDragAndDrop();
    initNavbarScroll();
    initMobileNav();
    initThemeToggle();
    initAnnNeuronVisualizer();
    
    // Load Model Info and Metrics from API (with Mock Fallback)
    fetchModelInfo();
    fetchPerformanceMetrics();
    
    // Setup UI Reset Handlers
    document.getElementById('resetBtn').addEventListener('click', resetLabSystem);
});

/* ==========================================================================
   GLOBAL VARIABLES & STATES
   ========================================================================== */
let drawCanvas, drawCtx;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let probabilityChart = null;
let uploadedFileBlob = null;
let isModelProcessing = false;

// API Endpoints
const API_PREDICT = '/predict';
const API_DRAW = '/draw';
const API_MODEL_INFO = '/model-info';
const API_METRICS = '/metrics';
const API_RESET = '/reset';

/* ==========================================================================
   STICKY NAVBAR & MOBILE TOGGLE
   ========================================================================== */
function initNavbarScroll() {
    const navbar = document.getElementById('navbar');
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', () => {
        // Sticky Glow Effect
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Active Link Highlight on Scroll
        let currentSectionId = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop - varNavbarHeight();
            const sectionHeight = section.clientHeight;
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                currentSectionId = section.getAttribute('id');
            }
        });

        if (currentSectionId) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${currentSectionId}`) {
                    link.classList.add('active');
                }
            });
        }
    });

    function varNavbarHeight() {
        return 90;
    }
}

function initMobileNav() {
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');
    
    navToggle.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        const icon = navToggle.querySelector('i');
        if (navLinks.classList.contains('active')) {
            icon.className = 'fa-solid fa-xmark';
        } else {
            icon.className = 'fa-solid fa-bars';
        }
    });

    // Close mobile nav when clicking a link
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('active');
            navToggle.querySelector('i').className = 'fa-solid fa-bars';
        });
    });
}

/* ==========================================================================
   3D PARTICLE BACKGROUND
   ========================================================================== */
function initParticleBackground() {
    const canvas = document.getElementById('particleCanvas');
    const ctx = canvas.getContext('2d');
    
    let particlesArray = [];
    const particleCount = 100;
    const connectionDistance = 120;
    
    const mouse = {
        x: null,
        y: null,
        radius: 140
    };
    
    window.addEventListener('mousemove', (event) => {
        mouse.x = event.clientX;
        mouse.y = event.clientY;
    });
    
    window.addEventListener('mouseout', () => {
        mouse.x = null;
        mouse.y = null;
    });
    
    window.addEventListener('resize', resizeCanvas);
    
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        particlesArray.forEach(p => p.clampPosition());
    }
    
    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2.5 + 1.2;
            this.speedX = (Math.random() - 0.5) * 0.35;
            this.speedY = (Math.random() - 0.5) * 0.35;
            this.color = Math.random() > 0.5 ? '#A855F7' : '#D8B4FE';
            this.pulseSpeed = Math.random() * 0.02 + 0.01;
            this.pulseAngle = Math.random() * Math.PI;
        }
        
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            this.pulseAngle += this.pulseSpeed;
            
            // Boundary bounce
            if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
            if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
            
            // Mouse Interaction: Smooth repulsion
            if (mouse.x != null && mouse.y != null) {
                let dx = this.x - mouse.x;
                let dy = this.y - mouse.y;
                let distance = Math.hypot(dx, dy);
                if (distance < mouse.radius) {
                    let force = (mouse.radius - distance) / mouse.radius;
                    let angle = Math.atan2(dy, dx);
                    this.x += Math.cos(angle) * force * 1.2;
                    this.y += Math.sin(angle) * force * 1.2;
                }
            }
        }
        
        draw() {
            const currentSize = Math.max(0.5, this.size + Math.sin(this.pulseAngle) * 0.4);
            ctx.beginPath();
            ctx.arc(this.x, this.y, currentSize, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.shadowBlur = 5;
            ctx.shadowColor = this.color;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
        
        clampPosition() {
            if (this.x > canvas.width) this.x = canvas.width;
            if (this.y > canvas.height) this.y = canvas.height;
        }
    }
    
    function init() {
        resizeCanvas();
        particlesArray = [];
        for (let i = 0; i < particleCount; i++) {
            particlesArray.push(new Particle());
        }
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw links first
        for (let a = 0; a < particlesArray.length; a++) {
            for (let b = a + 1; b < particlesArray.length; b++) {
                let dx = particlesArray[a].x - particlesArray[b].x;
                let dy = particlesArray[a].y - particlesArray[b].y;
                let distance = Math.hypot(dx, dy);
                
                if (distance < connectionDistance) {
                    let opacity = (1 - (distance / connectionDistance)) * 0.14;
                    // If mouse is close to both nodes, make the connection glow brighter
                    if (mouse.x != null && mouse.y != null) {
                        let mDist1 = Math.hypot(particlesArray[a].x - mouse.x, particlesArray[a].y - mouse.y);
                        let mDist2 = Math.hypot(particlesArray[b].x - mouse.x, particlesArray[b].y - mouse.y);
                        if (mDist1 < mouse.radius && mDist2 < mouse.radius) {
                            opacity *= 1.8;
                        }
                    }
                    ctx.strokeStyle = `rgba(168, 85, 247, ${opacity})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                    ctx.lineTo(particlesArray[b].x, particlesArray[b].y);
                    ctx.stroke();
                }
            }
            
            // Connect to mouse
            if (mouse.x != null && mouse.y != null) {
                let dx = particlesArray[a].x - mouse.x;
                let dy = particlesArray[a].y - mouse.y;
                let distance = Math.hypot(dx, dy);
                if (distance < mouse.radius) {
                    let opacity = (1 - (distance / mouse.radius)) * 0.22;
                    ctx.strokeStyle = `rgba(216, 180, 254, ${opacity})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.stroke();
                }
            }
        }
        
        // Update & Draw particles
        particlesArray.forEach(particle => {
            particle.update();
            particle.draw();
        });
        
        requestAnimationFrame(animate);
    }
    
    init();
    animate();
}

/* ==========================================================================
   HERO ANN SIGNAL VISUALIZER
   ========================================================================== */
function initHeroAnn() {
    const digitDisplay = document.getElementById('heroOutputDigit');
    const svgs = document.querySelectorAll('.bg-ann-svg, .hero-ann-svg, #heroAnnSvg');
    
    // 1. Output Digit Cycle Loop
    if (digitDisplay) {
        setInterval(() => {
            if (!isModelProcessing) {
                digitDisplay.textContent = Math.floor(Math.random() * 10);
                digitDisplay.classList.add('zoom-in');
                setTimeout(() => digitDisplay.classList.remove('zoom-in'), 300);
            }
        }, 1500);
    }

    if (svgs.length === 0) return;
    
    svgs.forEach(svg => {
        const paths = svg.querySelectorAll('.network-path');
        const particles = svg.querySelectorAll('.signal-particle');
        
        // 2. Dynamic Signal Particle Routing inside SVG
        particles.forEach((particle, idx) => {
            setInterval(() => {
                const randomPath = paths[Math.floor(Math.random() * paths.length)];
                const d = randomPath.getAttribute('d');
                particle.style.animation = 'none';
                particle.offsetHeight; /* Trigger Reflow */
                particle.style.offsetPath = `path("${d}")`;
                particle.style.animation = 'signal-travel 2.5s infinite linear';
            }, 3000 + (idx * 400));
        });
    });
}

/* ==========================================================================
   DRAWING CANVAS COMPONENT
   ========================================================================== */
function initDrawingCanvas() {
    drawCanvas = document.getElementById('drawingCanvas');
    drawCtx = drawCanvas.getContext('2d');
    
    const brushSlider = document.getElementById('brushSlider');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const clearBtn = document.getElementById('clearBtn');
    const predictDrawBtn = document.getElementById('predictDrawBtn');
    
    // Canvas Config
    drawCtx.fillStyle = '#000000';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    
    drawCtx.strokeStyle = '#FFFFFF';
    drawCtx.lineJoin = 'round';
    drawCtx.lineCap = 'round';
    drawCtx.lineWidth = parseInt(brushSlider.value);
    
    // Slider event
    brushSlider.addEventListener('input', (e) => {
        const size = e.target.value;
        brushSizeVal.textContent = `${size}px`;
        drawCtx.lineWidth = size;
    });

    // Drawing Events: Mouse
    drawCanvas.addEventListener('mousedown', startDrawing);
    drawCanvas.addEventListener('mousemove', draw);
    drawCanvas.addEventListener('mouseup', stopDrawing);
    drawCanvas.addEventListener('mouseleave', stopDrawing);

    // Drawing Events: Touch (Responsive Mobile support)
    drawCanvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = drawCanvas.getBoundingClientRect();
        lastX = touch.clientX - rect.left;
        lastY = touch.clientY - rect.top;
        isDrawing = true;
    });
    
    drawCanvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;
        const touch = e.touches[0];
        const rect = drawCanvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        
        drawCtx.beginPath();
        drawCtx.moveTo(lastX, lastY);
        drawCtx.lineTo(x, y);
        drawCtx.stroke();
        [lastX, lastY] = [x, y];
    });
    
    drawCanvas.addEventListener('touchend', stopDrawing);

    // Clear Button
    clearBtn.addEventListener('click', () => {
        drawCtx.fillStyle = '#000000';
        drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
        showToast('Info', 'Drawing canvas cleared.', 'success');
    });

    // Predict Drawn Button
    predictDrawBtn.addEventListener('click', submitDrawnDigit);

    function startDrawing(e) {
        isDrawing = true;
        const rect = drawCanvas.getBoundingClientRect();
        [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];
    }

    function draw(e) {
        if (!isDrawing) return;
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        drawCtx.beginPath();
        drawCtx.moveTo(lastX, lastY);
        drawCtx.lineTo(x, y);
        drawCtx.stroke();
        [lastX, lastY] = [x, y];
    }

    function stopDrawing() {
        isDrawing = false;
    }
}

/* ==========================================================================
   DRAG AND DROP / FILE SELECT COMPONENT
   ========================================================================== */
function initDragAndDrop() {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImgBtn = document.getElementById('removeImgBtn');
    const predictUploadBtn = document.getElementById('predictUploadBtn');

    // Trigger input click
    dropArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    // Drag-over highlights
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.classList.remove('dragover');
        }, false);
    });

    // Drop Files
    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    });

    // Clear Preview File
    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        uploadedFileBlob = null;
        previewContainer.style.display = 'none';
        dropArea.style.display = 'block';
        predictUploadBtn.disabled = true;
    });

    predictUploadBtn.addEventListener('click', submitUploadedImage);

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        // Validation
        if (!file.type.startsWith('image/')) {
            showToast('Format Error', 'Please upload image file formats only.', 'error');
            return;
        }

        uploadedFileBlob = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropArea.style.display = 'none';
            previewContainer.style.display = 'flex';
            predictUploadBtn.disabled = false;
            showToast('Success', 'Image loaded successfully.', 'success');
        };
        reader.readAsDataURL(file);
    }
}

/* ==========================================================================
   TAB CONTROLLER
   ========================================================================== */
window.switchInputTab = function(type) {
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(t => t.classList.remove('active'));
    contents.forEach(c => c.classList.remove('active'));

    if (type === 'draw') {
        tabs[0].classList.add('active');
        document.getElementById('drawTab').classList.add('active');
    } else {
        tabs[1].classList.add('active');
        document.getElementById('uploadTab').classList.add('active');
    }
};

/* ==========================================================================
   API CLIENTS & MODEL PIPELINE HANDLING
   ========================================================================== */

// 1. POST Draw Digit
async function submitDrawnDigit() {
    if (isModelProcessing) return;
    
    // Retrieve base64 string from canvas
    const base64Data = drawCanvas.toDataURL('image/png');
    
    startInferenceAnimation();
    
    try {
        const response = await fetch(API_DRAW, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Data })
        });
        
        if (!response.ok) throw new Error('API server response failed.');
        
        const data = await response.json();
        updatePredictionUI(data);
        showToast('Success', 'ANN processed drawing canvas.', 'success');
    } catch (err) {
        console.warn('API /draw failed. Activating local mock prediction model for validation...', err);
        generateMockPrediction(base64Data);
    } finally {
        stopInferenceAnimation();
    }
}

// 2. POST Uploaded Image
async function submitUploadedImage() {
    if (isModelProcessing || !uploadedFileBlob) return;
    
    const formData = new FormData();
    formData.append('image', uploadedFileBlob);
    
    startInferenceAnimation();
    
    try {
        const response = await fetch(API_PREDICT, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('API server response failed.');
        
        const data = await response.json();
        updatePredictionUI(data);
        showToast('Success', 'ANN processed uploaded image file.', 'success');
    } catch (err) {
        console.warn('API /predict failed. Loading local mock prediction model validation...', err);
        
        // Convert Blob to dataUrl for preprocessing grid mock view
        const reader = new FileReader();
        reader.onload = function(e) {
            generateMockPrediction(e.target.result);
        };
        reader.readAsDataURL(uploadedFileBlob);
    } finally {
        stopInferenceAnimation();
    }
}

// 3. GET Model Config Info
async function fetchModelInfo() {
    try {
        const response = await fetch(API_MODEL_INFO);
        if (!response.ok) throw new Error('Network offline');
        const data = await response.json();
        
        // Populate
        document.getElementById('modelOptimizer').textContent = data.optimizer || 'Adam';
        document.getElementById('modelLoss').textContent = data.loss_function || 'Categorical Crossentropy';
        if (data.accuracy) {
            document.getElementById('modelAccuracy').textContent = `${(data.accuracy).toFixed(2)}%`;
        }
        
        // Set layers list
        if (data.layers && Array.isArray(data.layers)) {
            populateArchitectureList(data.layers);
        } else if (data.architecture && Array.isArray(data.architecture)) {
            populateArchitectureList(data.architecture);
        }
    } catch (err) {
        console.warn('GET /model-info failed. Initializing default mockup variables.', err);
        // Defaults are hardcoded inside HTML, no actions needed.
    }
}

// 4. GET System Metrics
async function fetchPerformanceMetrics() {
    try {
        const response = await fetch(API_METRICS);
        if (!response.ok) throw new Error('Network offline');
        const data = await response.json();
        
        updateMetricsUI(data);
    } catch (err) {
        console.warn('GET /metrics failed. Populating standard model performance summaries...', err);
        loadMockMetrics();
    }
}

// 5. POST System Reset
async function resetLabSystem() {
    // Clear canvasses
    drawCtx.fillStyle = '#000000';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    
    // Clear upload file
    document.getElementById('fileInput').value = '';
    uploadedFileBlob = null;
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('dropArea').style.display = 'block';
    document.getElementById('predictUploadBtn').disabled = true;
    
    // Clear Results Panels
    document.getElementById('predDigit').textContent = '-';
    document.getElementById('predConfidenceText').textContent = '0.0%';
    document.getElementById('predConfidenceBar').style.width = '0%';
    document.getElementById('processingTime').textContent = '0 ms';
    
    document.getElementById('topPredictionsList').innerHTML = `<div class="top-pred-row empty-row">No input analyzed yet</div>`;
    document.getElementById('xaiExplanationText').textContent = 'Draw or upload a digit and execute prediction to generate structural descriptions. The explanation AI will analyze loops, strokes, contours, and node activations.';
    document.getElementById('whyNotList').innerHTML = `<div class="why-not-empty">Details will display alternate digit evaluations and activation barriers here.</div>`;
    
    // Clear preproc
    resetPreprocessingContainers();
    
    // Clear Chart.js
    if (probabilityChart) {
        probabilityChart.data.datasets[0].data = Array(10).fill(0);
        probabilityChart.update();
    }
    
    // Reset indicators
    resetJourneySteps();
    resetLayerGlows();

    // Reset ANN Neuron Panel
    const annBadge = document.getElementById('annStatusBadge');
    if (annBadge) { annBadge.textContent = 'Idle'; annBadge.className = 'ann-status-badge'; }
    const annWaiting = document.getElementById('annPredWaiting');
    const annOutput  = document.getElementById('annPredOutput');
    if (annWaiting) annWaiting.style.display = 'flex';
    if (annOutput)  annOutput.style.display  = 'none';
    const annSvg = document.getElementById('predAnnSvg');
    if (annSvg) {
        annSvg.querySelectorAll('.ann-node-circle').forEach(n => n.classList.remove('node-active','node-firing','output-winner'));
        annSvg.querySelectorAll('.ann-conn-group').forEach(g => g.classList.remove('conn-active'));
        document.getElementById('ann-signals').innerHTML = '';
    }
    resetAnnPhases();
    window._annNeuronAnimating = false;

    
    try {
        await fetch(API_RESET, { method: 'POST' });
        showToast('Reset Complete', 'Lab states refreshed.', 'success');
    } catch (e) {
        showToast('Local Reset', 'System states cleared.', 'success');
    }
}

/* ==========================================================================
   UI DATA POPULATION & ANIMATIONS
   ========================================================================== */

function populateArchitectureList(layers) {
    const list = document.getElementById('archLayerList');
    list.innerHTML = '';
    layers.forEach(layer => {
        const item = document.createElement('li');
        
        let badgeName = '';
        let extra = '';
        
        if (typeof layer === 'string') {
            // Handle string formats (mock/legacy)
            const nameParts = layer.split(' ');
            badgeName = nameParts[0];
            extra = nameParts.slice(1).join(' ');
        } else if (layer && typeof layer === 'object') {
            // Handle Keras layer objects from Flask API
            const rawName = layer.name || 'Layer';
            if (rawName.toLowerCase().includes('flatten')) {
                badgeName = 'Input Layer';
                extra = `(Flatten to 784 features)`;
            } else if (rawName.toLowerCase().includes('dense')) {
                badgeName = `Dense_${layer.units || ''}`;
                extra = `(${layer.activation ? layer.activation + ' activation' : ''})`;
            } else if (rawName.toLowerCase().includes('dropout')) {
                badgeName = 'Dropout';
                extra = `(${layer.rate ? 'rate ' + layer.rate : 'regularization'})`;
            } else {
                badgeName = rawName;
                extra = layer.activation ? `(${layer.activation})` : '';
            }
        } else {
            return;
        }
        
        item.innerHTML = `<span class="l-step">${badgeName}</span> ${extra}`;
        list.appendChild(item);
    });
}

function updateMetricsUI(data) {
    if (data.accuracy) document.getElementById('modelAccuracy').textContent = `${parseFloat(data.accuracy).toFixed(2)}%`;
    if (data.precision) document.getElementById('modelPrecision').textContent = `${parseFloat(data.precision).toFixed(2)}%`;
    if (data.recall) document.getElementById('modelRecall').textContent = `${parseFloat(data.recall).toFixed(2)}%`;
    if (data.f1_score) document.getElementById('modelF1').textContent = `${parseFloat(data.f1_score).toFixed(2)}%`;
    
    // Render Confusion Matrix
    if (data.confusion_matrix && data.confusion_matrix.length > 0) {
        renderConfusionMatrix(data.confusion_matrix);
    }
    
    // Render Wrong Predictions Gallery
    if (data.wrong_predictions && data.wrong_predictions.length > 0) {
        renderWrongGallery(data.wrong_predictions);
    }
}

function updatePredictionUI(data) {
    // Detect if we got the Pixel2Digit API structure or the original mock format
    const isP2D = data.prediction !== undefined;
    
    // 1. Numeric Values
    const digit = isP2D ? data.prediction.digit : data.predicted_digit;
    const confidence = parseFloat(isP2D ? data.prediction.confidence : data.confidence).toFixed(1);
    
    const predDigitEl = document.getElementById('predDigit');
    predDigitEl.textContent = digit !== undefined ? digit : '-';
    predDigitEl.classList.add('zoom-in');
    setTimeout(() => predDigitEl.classList.remove('zoom-in'), 500);
    
    document.getElementById('predConfidenceText').textContent = `${confidence}%`;
    document.getElementById('predConfidenceBar').style.width = `${confidence}%`;
    
    let timeText = '0 ms';
    if (isP2D) {
        timeText = typeof data.prediction.prediction_time === 'number' 
            ? `${data.prediction.prediction_time} ms` 
            : data.prediction.prediction_time;
    } else {
        timeText = `${data.processing_time || 0} ms`;
    }
    document.getElementById('processingTime').textContent = timeText;

    // 2. Synchronize Hero Output digit
    const heroOutput = document.getElementById('heroOutputDigit');
    heroOutput.textContent = digit !== undefined ? digit : '-';
    heroOutput.classList.add('zoom-in');
    setTimeout(() => heroOutput.classList.remove('zoom-in'), 300);

    // 3. Top Predictions list
    const topPredsBox = document.getElementById('topPredictionsList');
    topPredsBox.innerHTML = '';
    
    const predictions = isP2D ? data.top3 : data.top_predictions;
    if (predictions && predictions.length > 0) {
        predictions.slice(0, 3).forEach((pred, rank) => {
            const row = document.createElement('div');
            row.className = 'top-pred-row';
            row.innerHTML = `
                <div class="rank-digit">
                    <span class="rank-num">#${rank+1}</span>
                    <span class="digit-val">Digit ${pred.digit}</span>
                </div>
                <span class="prob-val">${parseFloat(pred.probability).toFixed(1)}%</span>
            `;
            topPredsBox.appendChild(row);
        });
    } else {
        topPredsBox.innerHTML = `<div class="top-pred-row empty-row">No input analyzed yet</div>`;
    }

    // 4. Explanation text
    const explanationText = isP2D ? data.xai.why_this_digit : data.explanation;
    document.getElementById('xaiExplanationText').innerHTML = (explanationText || 'No structural explanation generated.').replace(/\n/g, '<br>');

    // 5. Why Not List
    const whyNotBox = document.getElementById('whyNotList');
    whyNotBox.innerHTML = '';
    const alternateDigits = isP2D ? data.xai.why_not_others : data.why_not;
    if (!alternateDigits || alternateDigits.length === 0) {
        whyNotBox.innerHTML = `<div class="why-not-empty">All alternate digit evaluations succeeded. No anomalies.</div>`;
    } else {
        const listWrap = document.createElement('div');
        listWrap.className = 'why-not-list-container';
        alternateDigits.forEach(item => {
            const div = document.createElement('div');
            div.className = 'why-not-item';
            div.innerHTML = `<span class="why-not-digit">Not ${item.digit}:</span><span class="why-not-reason">${item.reason}</span>`;
            listWrap.appendChild(div);
        });
        whyNotBox.appendChild(listWrap);
    }

    // 6. Probability Chart.js update
    let probsList = [];
    if (isP2D) {
        // all_probabilities is array of {"digit": i, "probability": float}
        // Let's sort it by digit order 0 to 9
        const sortedProbs = [...data.all_probabilities].sort((a, b) => a.digit - b.digit);
        probsList = sortedProbs.map(p => p.probability);
    } else {
        probsList = data.probabilities || [];
    }
    updateChart(probsList);

    // 7. Preprocessing Images View
    let preprocData = {};
    if (isP2D) {
        const formatB64 = (b64) => {
            if (!b64) return null;
            return b64.startsWith('data:image') ? b64 : `data:image/png;base64,${b64}`;
        };
        
        preprocData = {
            original: formatB64(data.preprocessing_steps.original),
            grayscale: formatB64(data.preprocessing_steps.grayscale),
            resized: formatB64(data.preprocessing_steps.resized),
            normalized: formatB64(data.preprocessing_steps.normalized),
            pixel_grid: data.pixel_grid && data.pixel_grid.pixels ? data.pixel_grid.pixels.flat() : []
        };
        
        // If original/grayscale is null (canvas drawing predict), fallback to original drawing canvas image
        if (!preprocData.original && drawCanvas) {
            const canvasImg = drawCanvas.toDataURL('image/png');
            preprocData.original = canvasImg;
            preprocData.grayscale = canvasImg;
        }
    } else {
        preprocData = data.preprocessing || {};
    }
    updatePreprocessingImages(preprocData);
    
    // 8. Completed pipeline states
    animatePipelineSuccess();
    
    // 9. Glow architecture output layer
    highlightLayer('output');

    // 10. Animate ANN Neuron Panel
    const digitNum = digit !== undefined ? parseInt(digit) : 0;
    const confNum  = confidence !== undefined ? parseFloat(confidence) : 0;
    animateAnnNeuronPanel(digitNum, confNum);
}

// Chart.js render function
function updateChart(probabilities) {
    // Normalizer to sum up to 100 or format to percents
    const dataPoints = probabilities.map(val => parseFloat(val).toFixed(2));
    
    if (probabilityChart) {
        probabilityChart.data.datasets[0].data = dataPoints;
        probabilityChart.update();
    } else {
        const ctx = document.getElementById('probabilityChart').getContext('2d');
        
        // Define Gradient Color
        const gradient = ctx.createLinearGradient(0, 0, 300, 0);
        gradient.addColorStop(0, '#7C3AED'); // Deep Lavender
        gradient.addColorStop(1, '#C084FC'); // Light Lavender
        
        probabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                datasets: [{
                    label: 'Probability (%)',
                    data: dataPoints,
                    backgroundColor: gradient,
                    borderColor: '#7C3AED',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y', // Horizontal bars
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#FFFFFF',
                        titleColor: '#1E1035',
                        bodyColor: '#7C3AED',
                        borderColor: 'rgba(139, 92, 246, 0.2)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(139, 92, 246, 0.1)' },
                        ticks: { color: '#6B7280' },
                        min: 0,
                        max: 100
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#3B1F6B', font: { weight: 'bold' } }
                    }
                }
            }
        });
    }
}

// Confusion matrix population
function renderConfusionMatrix(matrix) {
    const tbody = document.getElementById('confusionMatrixBody');
    tbody.innerHTML = '';
    
    for (let r = 0; r < 10; r++) {
        const tr = document.createElement('tr');
        
        // Row header label
        const rowHeader = document.createElement('th');
        rowHeader.textContent = r;
        tr.appendChild(rowHeader);
        
        const rowData = matrix[r] || Array(10).fill(0);
        
        // Row sums for relative brightness opacity
        const rowSum = rowData.reduce((a, b) => a + b, 0) || 1;
        
        for (let c = 0; c < 10; c++) {
            const td = document.createElement('td');
            const val = rowData[c];
            td.textContent = val;
            
            // Apply gradient scale: Diagonal is hits, else misclassifications
            if (r === c) {
                // High accuracy hits (Green/cyan gradient)
                const hitRatio = val / rowSum;
                td.style.backgroundColor = `rgba(34, 197, 94, ${hitRatio.toFixed(2)})`;
                td.style.color = hitRatio > 0.5 ? '#000000' : '#FFFFFF';
                td.style.fontWeight = 'bold';
            } else if (val > 0) {
                // False positives (red background glow scale)
                const errorRatio = Math.min(val / (rowSum * 0.1), 1); // scale sensitivity
                td.style.backgroundColor = `rgba(239, 68, 68, ${(errorRatio * 0.6).toFixed(2)})`;
                td.style.color = '#FFFFFF';
            }
            
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    }
}

// Wrong predictions layout
function renderWrongGallery(wrongList) {
    const grid = document.getElementById('wrongPredictionsGrid');
    grid.innerHTML = '';
    
    wrongList.forEach(item => {
        const actual = item.actual_digit !== undefined ? item.actual_digit : item.true_label;
        const predicted = item.predicted_digit !== undefined ? item.predicted_digit : item.predicted_label;
        const confidence = parseFloat(item.confidence).toFixed(1);
        
        let imgSrc = '';
        if (item.image_url) {
            imgSrc = item.image_url;
        } else if (item.image) {
            imgSrc = item.image.startsWith('data:image') ? item.image : `data:image/png;base64,${item.image}`;
        }
        
        const explanation = item.explanation || `The neural network misclassified this digit ${actual} as ${predicted} due to structural similarities or skewed lines.`;
        
        const card = document.createElement('div');
        card.className = 'wrong-card';
        card.innerHTML = `
            <div class="wrong-img-box">
                <img src="${imgSrc}" alt="Wrong Digit Actual ${actual}">
            </div>
            <div class="wrong-meta-container">
                <span class="wrong-meta">Actual: <span class="wrong-actual">${actual}</span></span>
                <span class="wrong-meta">Pred: <span class="wrong-pred">${predicted}</span></span>
                <span class="wrong-conf">${confidence}%</span>
            </div>
            <p class="wrong-desc">${explanation}</p>
        `;
        grid.appendChild(card);
    });
}

// Toggle pipeline stages sequentially (1-by-1) during computation simulation
async function simulatePipelineStages() {
    resetJourneySteps();
    resetLayerGlows();
    
    const sleep = (ms) => new Promise(res => setTimeout(res, ms));
    
    // Step 1: Image Received
    highlightJourneyStep(1, 'processing');
    highlightLayer('input');
    await sleep(200);
    highlightJourneyStep(1, 'success');
    
    // Step 2: Grayscale conversion
    highlightJourneyStep(2, 'processing');
    await sleep(150);
    highlightJourneyStep(2, 'success');
    
    // Step 3: Resizing
    highlightJourneyStep(3, 'processing');
    await sleep(150);
    highlightJourneyStep(3, 'success');
    
    // Step 4: Normalizing
    highlightJourneyStep(4, 'processing');
    await sleep(120);
    highlightJourneyStep(4, 'success');
    
    // Step 5: Flattening
    highlightJourneyStep(5, 'processing');
    highlightLayer('dense1');
    await sleep(100);
    highlightJourneyStep(5, 'success');
    
    // Step 6: ANN Propagation
    highlightJourneyStep(6, 'processing');
    highlightLayer('dense2');
    highlightLayer('dropout');
    await sleep(250);
    highlightJourneyStep(6, 'success');
    
    // Step 7: Output Probabilities
    highlightJourneyStep(7, 'processing');
    highlightLayer('dense3');
    await sleep(150);
    highlightJourneyStep(7, 'success');
    
    // Step 8: Decoded prediction explanation
    highlightJourneyStep(8, 'processing');
}

function highlightJourneyStep(stepNum, status) {
    const stepEl = document.getElementById(`step-${stepNum}`);
    stepEl.classList.remove('step-processing', 'step-success');
    if (status === 'processing') stepEl.classList.add('step-processing');
    if (status === 'success') stepEl.classList.add('step-success');
}

function animatePipelineSuccess() {
    highlightJourneyStep(8, 'success');
}

function resetJourneySteps() {
    for (let i = 1; i <= 8; i++) {
        const step = document.getElementById(`step-${i}`);
        step.classList.remove('step-processing', 'step-success');
    }
}

function highlightLayer(layerName) {
    const indicators = document.querySelectorAll('.layer-indicator');
    indicators.forEach(ind => {
        if (ind.dataset.layer === layerName) {
            ind.classList.add('glow-active');
        }
    });
}

function resetLayerGlows() {
    const indicators = document.querySelectorAll('.layer-indicator');
    indicators.forEach(ind => ind.classList.remove('glow-active'));
}

function startInferenceAnimation() {
    isModelProcessing = true;
    simulatePipelineStages();
}

function stopInferenceAnimation() {
    isModelProcessing = false;
}

// Populate images under preprocessing dashboard
function updatePreprocessingImages(preproc) {
    updatePreprocCardBox('preproc-original-box', preproc.original, 'fa-image', 'Raw Input image');
    updatePreprocCardBox('preproc-grayscale-box', preproc.grayscale, 'fa-fill-drip', 'Grayscale single-channel');
    updatePreprocCardBox('preproc-resized-box', preproc.resized, 'fa-minimize', 'Downscaled 28x28 matrix');
    updatePreprocCardBox('preproc-normalized-box', preproc.normalized, 'fa-calculator', 'Scaled floats [0.0, 1.0]');
    
    // Update Heatmap grids
    if (preproc.pixel_grid && preproc.pixel_grid.length > 0) {
        populatePixelHeatmaps(preproc.pixel_grid);
    }
}

function updatePreprocCardBox(containerId, dataSrc, fallbackIcon, altLabel) {
    const box = document.getElementById(containerId);
    if (!dataSrc) {
        box.innerHTML = `<span class="preproc-placeholder"><i class="fa-solid ${fallbackIcon}"></i> Empty</span>`;
        return;
    }
    
    // Check if base64/URL or matrix string
    if (typeof dataSrc === 'string' && (dataSrc.startsWith('data:image') || dataSrc.includes('/') || dataSrc.startsWith('http'))) {
        box.innerHTML = `<img src="${dataSrc}" alt="${altLabel}">`;
    } else {
        // Fallback placeholder display
        box.innerHTML = `<span class="preproc-placeholder"><i class="fa-solid ${fallbackIcon}"></i> ${dataSrc.substring(0, 15)}...</span>`;
    }
}

function resetPreprocessingContainers() {
    const clearPlaceholder = (id, icon) => {
        document.getElementById(id).innerHTML = `<span class="preproc-placeholder"><i class="fa-solid ${icon}"></i> Waiting for digit...</span>`;
    };
    clearPlaceholder('preproc-original-box', 'fa-image');
    clearPlaceholder('preproc-grayscale-box', 'fa-fill-drip');
    clearPlaceholder('preproc-resized-box', 'fa-minimize');
    clearPlaceholder('preproc-normalized-box', 'fa-calculator');
    
    // Clear mini and large grids
    document.getElementById('miniHeatmapGrid').innerHTML = '';
    document.getElementById('preprocHeatmapGrid').innerHTML = '';
    document.getElementById('heatmapCellVal').textContent = 'Cell Value: - | Cell ID: -';
}

function populatePixelHeatmaps(flatArray) {
    const miniGrid = document.getElementById('miniHeatmapGrid');
    const largeGrid = document.getElementById('preprocHeatmapGrid');
    const cellDisplay = document.getElementById('heatmapCellVal');
    
    miniGrid.innerHTML = '';
    largeGrid.innerHTML = '';
    
    flatArray.forEach((val, idx) => {
        const floatVal = parseFloat(val).toFixed(2);
        
        // Intensity scaling color
        // Mapped to RGB shades of Cyan #00F5FF (R=0, G=245, B=255)
        const gChannel = Math.floor(245 * floatVal);
        const bChannel = Math.floor(255 * floatVal);
        
        // CSS Color
        const cellColor = floatVal > 0 ? `rgba(0, ${gChannel}, ${bChannel}, ${Math.max(floatVal, 0.25)})` : '#070B1F';
        
        // 1. Mini grid cell
        const mCell = document.createElement('div');
        mCell.className = 'heatmap-cell';
        mCell.style.backgroundColor = cellColor;
        miniGrid.appendChild(mCell);
        
        // 2. Large grid cell
        const lCell = document.createElement('div');
        lCell.className = 'preproc-heatmap-cell';
        lCell.style.backgroundColor = cellColor;
        
        // Hover details tracking
        lCell.addEventListener('mouseenter', () => {
            cellDisplay.textContent = `Cell Value: ${floatVal} | Cell ID: X[${idx}]`;
            cellDisplay.style.color = floatVal > 0 ? '#00F5FF' : '#94A3B8';
        });
        
        largeGrid.appendChild(lCell);
    });
}

/* ==========================================================================
   SELF-HEALING HARDCODED FALLBACKS / HACKATHON OFFLINE SIMULATION
   ========================================================================== */

// Simulated model explanation texts database
const DIGIT_EXPLANATIONS = {
    0: "The neural network predicted 0 due to a single large, circular continuous border contour surrounding an empty central cavity. There is high activation in the diagonal boundary nodes, with no central vertical strokes detected.",
    1: "The model classified the digit as 1 because of a dominant, narrow vertical column vector with negligible width density. The pixel grid registers high intensity strokes solely down the vertical median, triggering zero closed-loop activation paths.",
    2: "A prediction of 2 is generated because the stroke features a smooth curved arc in the upper hemisphere, transitioning down a diagonal slope to a high-density, horizontal anchor line at the base.",
    3: "The digit is classified as 3 because the contours present two open, backward-facing loop components. The top arc curves back inwards toward the median, and the bottom arc wraps around without sealing any enclosed pixel regions.",
    4: "The neural network predicted 4 based on two intersecting vertical strokes connected by a horizontal spacer stroke in the lower half. The stroke forms a closed triangular or open cup structure at the top, combined with high vertical weight values.",
    5: "The model predicted 5 because the stroke starts with a horizontal header, drops vertically to the left, and sweeps into a wide, open lower-right arc segment.",
    6: "A classification of 6 is triggered by a continuous, downward-sloping curve that transitions into a fully enclosed loop region in the lower hemisphere. The top segment remains open and free of overlapping lines.",
    7: "The digit is predicted to be 7 because of a horizontal high-intensity border stroke at the top, meeting a sharp vertex that angles diagonally downwards toward the central footer coordinates.",
    8: "The neural network predicted 8 because the digit contains two fully enclosed circular loop-like regions stacked vertically. The central intersection point displays high stroke intensity, creating activation peaks in loops 1 and 2.",
    9: "The model predicted 9 due to an enclosed circular loop in the upper hemisphere combined with a downward vertical tail stroke stretching into the lower quadrant."
};

const ALT_DIGIT_CONTRASTS = {
    0: [
        { digit: 8, reason: "The central canvas pixels remain completely empty, blocking double-loop activations." },
        { digit: 6, reason: "The top region is symmetrical and fully balanced, lacking an open header stroke." }
    ],
    1: [
        { digit: 7, reason: "No horizontal top stroke is present to cross the vertical path." },
        { digit: 4, reason: "The stroke does not branch out into side components or intersecting joints." }
    ],
    2: [
        { digit: 7, reason: "The baseline contains a thick horizontal stroke, whereas 7 terminates at a single point." },
        { digit: 3, reason: "The lower region extends leftward to form a base rather than curving inward to the left." }
    ],
    3: [
        { digit: 8, reason: "The central crossing points do not loop back to seal the upper and lower cavities." },
        { digit: 9, reason: "The upper loop is open on the left, failing the top circle filter bounds." }
    ],
    4: [
        { digit: 9, reason: "The upper cup remains open at the top border, and there is no curved connection." },
        { digit: 1, reason: "Multiple non-adjacent horizontal strokes prevent it from matching a single line vector." }
    ],
    5: [
        { digit: 6, reason: "The vertical drop stroke on the left does not merge to seal a bottom loop." },
        { digit: 3, reason: "The upper horizontal segment is flat and rigid rather than wrapping in a curve." }
    ],
    6: [
        { digit: 8, reason: "The upper tail is completely open, preventing a double loop classification." },
        { digit: 0, reason: "The stroke splits into two distinct hemispheres: a closed loop and an open extension." }
    ],
    7: [
        { digit: 1, reason: "The horizontal cap has too much width and mass to be a straight vertical line." },
        { digit: 9, reason: "No closed loop is present at the top header, indicating open terminal vertices." }
    ],
    8: [
        { digit: 3, reason: "The outer loops are closed on both sides, which breaks the open-curve profile of a 3." },
        { digit: 0, reason: "The horizontal cross-over bar at the center divides the canvas into two smaller regions." }
    ],
    9: [
        { digit: 8, reason: "The lower tail is open, failing the double-loop requirements of the model." },
        { digit: 7, reason: "The upper loop is fully enclosed, separating it from the open vertex of a 7." }
    ]
};

// Simulated prediction engine for offline mode
function generateMockPrediction(dataSrcUrl) {
    // Generate a random prediction digit for visualization
    const mockDigit = Math.floor(Math.random() * 10);
    const mockConfidence = (Math.random() * 8) + 91.5; // 91.5% - 99.5%
    
    // Generate probabilities list
    const probs = Array(10).fill(0).map((_, i) => {
        if (i === mockDigit) return mockConfidence;
        return (100 - mockConfidence) * (Math.random() / 3);
    });
    
    // Clean probabilities sums
    const sum = probs.reduce((a,b) => a+b, 0);
    const normalizedProbs = probs.map(p => (p / sum) * 100);
    
    // Sort top predictions
    const topPreds = normalizedProbs.map((p, idx) => ({ digit: idx, probability: p }))
                                   .sort((a,b) => b.probability - a.probability);

    // Dynamic preproc image generation using hidden canvas
    // We can downscale the user input to 28x28 dynamically to create a REAL preview!
    const testImg = new Image();
    testImg.onload = function() {
        const hidCanvas = document.createElement('canvas');
        hidCanvas.width = 28;
        hidCanvas.height = 28;
        const hidCtx = hidCanvas.getContext('2d');
        
        // Draw image downscaled
        hidCtx.drawImage(testImg, 0, 0, 28, 28);
        const data28 = hidCtx.getImageData(0, 0, 28, 28);
        
        // Grayscale conversion
        const grayscaleData = hidCtx.createImageData(28, 28);
        const flatArray = [];
        
        for (let i = 0; i < data28.data.length; i += 4) {
            const r = data28.data[i];
            const g = data28.data[i+1];
            const b = data28.data[i+2];
            const a = data28.data[i+3];
            
            // Grayscale average weighted
            let v = (0.299 * r + 0.587 * g + 0.114 * b);
            
            // Adjust alpha weights
            if (a < 50) v = 0;
            
            grayscaleData.data[i] = v;
            grayscaleData.data[i+1] = v;
            grayscaleData.data[i+2] = v;
            grayscaleData.data[i+3] = 255;
            
            // Flat normalized array
            flatArray.push(v / 255);
        }
        
        hidCtx.putImageData(grayscaleData, 0, 0);
        const grayscaleBase64 = hidCanvas.toDataURL();
        
        // Populate preprocessed steps
        const mockResult = {
            predicted_digit: mockDigit,
            confidence: mockConfidence,
            top_predictions: topPreds,
            probabilities: normalizedProbs,
            processing_time: Math.floor((Math.random() * 50) + 30),
            explanation: DIGIT_EXPLANATIONS[mockDigit],
            why_not: ALT_DIGIT_CONTRASTS[mockDigit],
            preprocessing: {
                original: dataSrcUrl,
                grayscale: grayscaleBase64,
                resized: grayscaleBase64,
                normalized: grayscaleBase64,
                pixel_grid: flatArray
            }
        };
        
        updatePredictionUI(mockResult);
    };
    testImg.src = dataSrcUrl;
}

// Simulated Metrics database loader for offline support
function loadMockMetrics() {
    // 10x10 MNIST validation confusion matrix
    const matrix = [
        [970, 0, 2, 1, 1, 2, 2, 1, 1, 0],
        [0, 1125, 3, 1, 0, 1, 2, 1, 2, 0],
        [6, 3, 990, 8, 4, 0, 3, 8, 9, 1],
        [0, 0, 5, 985, 0, 10, 0, 4, 4, 2],
        [1, 1, 3, 0, 955, 0, 4, 2, 2, 14],
        [4, 1, 0, 12, 2, 855, 6, 2, 7, 3],
        [7, 3, 2, 1, 4, 6, 930, 0, 5, 0],
        [1, 5, 10, 4, 2, 0, 0, 992, 2, 12],
        [4, 4, 4, 8, 5, 6, 4, 2, 932, 5],
        [3, 3, 1, 7, 10, 2, 1, 8, 5, 969]
    ];

    // Wrong predictions sample database
    const wrongList = [
        {
            id: 101,
            image_url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByg4iAAAAAAElFTkSuQmCC",
            actual_digit: 4,
            predicted_digit: 9,
            confidence: 76.5,
            explanation: "The horizontal stroke was drawn extremely high, which closed the upper region and resembled a 9 circle."
        },
        {
            id: 102,
            image_url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByg4iAAAAAAElFTkSuQmCC",
            actual_digit: 5,
            predicted_digit: 3,
            confidence: 68.2,
            explanation: "The left corner drop was curved, bypassing the right-angle filter and matching a 3 trajectory."
        },
        {
            id: 103,
            image_url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByg4iAAAAAAElFTkSuQmCC",
            actual_digit: 7,
            predicted_digit: 1,
            confidence: 84.1,
            explanation: "The horizontal top bar was extremely short, causing features to align closely with standard slanted 1 vectors."
        },
        {
            id: 104,
            image_url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByg4iAAAAAAElFTkSuQmCC",
            actual_digit: 8,
            predicted_digit: 0,
            confidence: 59.4,
            explanation: "The central cross-over strokes did not overlap in the middle, leaving a single wide cavity open like a 0."
        }
    ];

    // Load Mock canvas drawings for the gallery programmatically
    const loadMockImages = () => {
        const colors = ['#4,9', '#5,3', '#7,1', '#8,0'];
        const digits = [4, 5, 7, 8];
        const preds = [9, 3, 1, 0];
        
        wrongList.forEach((item, idx) => {
            const canvas = document.createElement('canvas');
            canvas.width = 56;
            canvas.height = 56;
            const ctx = canvas.getContext('2d');
            
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, 56, 56);
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
            
            // Draw dummy slanted lines to look like raw digits
            ctx.beginPath();
            if (idx === 0) { // 4 -> 9
                ctx.moveTo(15, 15); ctx.lineTo(15, 35); ctx.lineTo(40, 35);
                ctx.moveTo(35, 15); ctx.lineTo(35, 45);
                ctx.moveTo(15, 15); ctx.lineTo(35, 15); // closing top loop like 9
            } else if (idx === 1) { // 5 -> 3
                ctx.moveTo(40, 15); ctx.lineTo(20, 15); ctx.lineTo(20, 30);
                ctx.bezierCurveTo(30, 30, 42, 35, 40, 45);
                ctx.bezierCurveTo(38, 52, 20, 52, 15, 45);
            } else if (idx === 2) { // 7 -> 1
                ctx.moveTo(25, 15); ctx.lineTo(38, 15); ctx.lineTo(20, 48);
            } else { // 8 -> 0
                ctx.arc(28, 28, 18, 0, Math.PI * 2);
            }
            ctx.stroke();
            item.image_url = canvas.toDataURL();
        });
    };
    
    loadMockImages();

    const mockData = {
        accuracy: 98.50,
        precision: 98.20,
        recall: 98.10,
        f1_score: 98.15,
        confusion_matrix: matrix,
        wrong_predictions: wrongList
    };

    updateMetricsUI(mockData);
}

/* ==========================================================================
   TOAST NOTIFICATION COMPONENT
   ========================================================================== */
function showToast(title, message, type = 'success') {
    const container = document.getElementById('toastContainer');
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = type === 'success' ? 'fa-circle-check' : 'fa-circle-exclamation';
    
    toast.innerHTML = `
        <div class="toast-icon"><i class="fa-solid ${icon}"></i></div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <div class="toast-close"><i class="fa-solid fa-xmark"></i></div>
    `;
    
    // Close button event
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.style.transform = 'translateX(120%)';
        setTimeout(() => toast.remove(), 300);
    });
    
    container.appendChild(toast);
    
    // Auto remove after 4.5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.style.transform = 'translateX(120%)';
            setTimeout(() => toast.remove(), 300);
        }
    }, 4500);
}

/* ==========================================================================
   LIGHT MODE TOGGLE
   ========================================================================== */
function initThemeToggle() {
    const btn = document.getElementById('themeToggleBtn');
    const icon = document.getElementById('themeIcon');
    if (!btn) return;

    // Restore saved preference
    const saved = localStorage.getItem('neurovision-theme');
    if (saved === 'light') {
        document.body.classList.add('light-mode');
        icon.className = 'fa-solid fa-moon';
    }

    btn.addEventListener('click', () => {
        const isLight = document.body.classList.toggle('light-mode');
        if (isLight) {
            icon.className = 'fa-solid fa-moon';
            localStorage.setItem('neurovision-theme', 'light');
            showToast('Light Mode', 'Switched to light theme.', 'success');
        } else {
            icon.className = 'fa-solid fa-sun';
            localStorage.setItem('neurovision-theme', 'dark');
            showToast('Dark Mode', 'Switched to dark theme.', 'success');
        }
    });
}

/* ==========================================================================
   ANN NEURON VISUALIZER
   ========================================================================== */
function initAnnNeuronVisualizer() {
    // Hook into prediction pipeline — wrap startInferenceAnimation
    const origStart = window.startInferenceAnimation || startInferenceAnimation;
    const origStop  = window.stopInferenceAnimation  || stopInferenceAnimation;

    window._annNeuronAnimating = false;
}

/**
 * Called when a prediction result arrives — runs the full layered animation
 * on the neuron panel SVG and then shows the predicted digit.
 */
async function animateAnnNeuronPanel(predictedDigit, confidence) {
    if (window._annNeuronAnimating) return;
    window._annNeuronAnimating = true;

    const badge   = document.getElementById('annStatusBadge');
    const waiting = document.getElementById('annPredWaiting');
    const output  = document.getElementById('annPredOutput');
    const svg     = document.getElementById('predAnnSvg');
    if (!svg) return;

    // Reset all node states
    svg.querySelectorAll('.ann-node-circle').forEach(n => {
        n.classList.remove('node-active', 'node-firing', 'output-winner');
    });
    svg.querySelectorAll('.ann-conn-group').forEach(g => g.classList.remove('conn-active'));
    resetAnnPhases();

    // Show predicting badge
    badge.textContent = 'Predicting...';
    badge.className = 'ann-status-badge predicting';
    waiting.style.display = 'flex';
    output.style.display  = 'none';

    const sleep = ms => new Promise(r => setTimeout(r, ms));

    // ── PHASE 1: Input layer fires ──────────────────────────────────────────
    setAnnPhase('phase-input', true);
    const inputNodes = svg.querySelectorAll('#layer-input .ann-node-circle');
    inputNodes.forEach((n, i) => {
        setTimeout(() => n.classList.add('node-active'), i * 60);
    });
    await spawnSignals(svg, 'conn-in-h1', 5, '#A855F7');
    await sleep(300);

    // ── PHASE 2: Hidden layer 1 fires ───────────────────────────────────────
    setAnnPhase('phase-h1', true);
    document.getElementById('conn-in-h1').classList.add('conn-active');
    const h1Nodes = svg.querySelectorAll('#layer-h1 .ann-node-circle');
    h1Nodes.forEach((n, i) => {
        setTimeout(() => {
            n.classList.remove('node-active');
            n.classList.add('node-firing');
            setTimeout(() => n.classList.remove('node-firing'), 400);
        }, i * 55);
    });
    await spawnSignals(svg, 'conn-h1-h2', 4, '#8B5CF6');
    await sleep(300);

    // ── PHASE 3: Hidden layer 2 fires ───────────────────────────────────────
    setAnnPhase('phase-h2', true);
    document.getElementById('conn-h1-h2').classList.add('conn-active');
    const h2Nodes = svg.querySelectorAll('#layer-h2 .ann-node-circle');
    h2Nodes.forEach((n, i) => {
        setTimeout(() => {
            n.classList.add('node-firing');
            setTimeout(() => n.classList.remove('node-firing'), 380);
        }, i * 70);
    });
    await spawnSignals(svg, 'conn-h2-h3', 3, '#7C3AED');
    await sleep(280);

    // ── PHASE 4: Hidden layer 3 fires ───────────────────────────────────────
    setAnnPhase('phase-h3', true);
    document.getElementById('conn-h2-h3').classList.add('conn-active');
    const h3Nodes = svg.querySelectorAll('#layer-h3 .ann-node-circle');
    h3Nodes.forEach((n, i) => {
        setTimeout(() => {
            n.classList.add('node-firing');
            setTimeout(() => n.classList.remove('node-firing'), 350);
        }, i * 80);
    });
    await spawnSignals(svg, 'conn-h3-out', 4, '#6D28D9');
    await sleep(280);

    // ── PHASE 5: Output layer — highlight winning node ──────────────────────
    setAnnPhase('phase-output', true);
    document.getElementById('conn-h3-out').classList.add('conn-active');
    const outputNodes = svg.querySelectorAll('#layer-output .ann-node-circle');
    outputNodes.forEach((n, i) => {
        const digit = parseInt(n.getAttribute('data-digit'));
        setTimeout(() => {
            if (digit === predictedDigit) {
                n.classList.add('output-winner');
            } else {
                n.classList.add('node-active');
                setTimeout(() => n.classList.remove('node-active'), 500);
            }
        }, i * 40);
    });

    await sleep(600);

    // ── Show result ──────────────────────────────────────────────────────────
    badge.textContent = 'Done ✓';
    badge.className = 'ann-status-badge done';

    document.getElementById('annPredDigitVal').textContent  = predictedDigit;
    document.getElementById('annPredDigitConf').textContent = `${parseFloat(confidence).toFixed(1)}%`;
    document.getElementById('annNodeIdx').textContent = predictedDigit;

    waiting.style.display = 'none';
    output.style.display  = 'flex';

    window._annNeuronAnimating = false;
}

/** Sets a phase label as active/inactive */
function setAnnPhase(phaseId, active) {
    const el = document.getElementById(phaseId);
    if (!el) return;
    if (active) el.classList.add('phase-active');
    else        el.classList.remove('phase-active');
}

function resetAnnPhases() {
    ['phase-input','phase-h1','phase-h2','phase-h3','phase-output'].forEach(id => setAnnPhase(id, false));
}

/**
 * Spawns animated signal dots that travel along the connection lines of a group.
 * @param {SVGElement} svg
 * @param {string} groupId    - e.g. 'conn-in-h1'
 * @param {number} count      - how many signals to spawn
 * @param {string} color      - signal dot color
 */
async function spawnSignals(svg, groupId, count, color) {
    const group = svg.getElementById ? svg.getElementById(groupId) : document.getElementById(groupId);
    if (!group) return;
    const lines = Array.from(group.querySelectorAll('line'));
    if (lines.length === 0) return;

    const signalContainer = document.getElementById('ann-signals');
    const duration = 450; // ms

    // Pick `count` random lines to animate
    const chosen = [];
    for (let i = 0; i < count; i++) {
        chosen.push(lines[Math.floor(Math.random() * lines.length)]);
    }

    const promises = chosen.map((line, idx) => new Promise(resolve => {
        const x1 = parseFloat(line.getAttribute('x1'));
        const y1 = parseFloat(line.getAttribute('y1'));
        const x2 = parseFloat(line.getAttribute('x2'));
        const y2 = parseFloat(line.getAttribute('y2'));

        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('r', '4');
        dot.setAttribute('fill', color);
        dot.setAttribute('filter', 'url(#glow)');
        dot.setAttribute('cx', x1);
        dot.setAttribute('cy', y1);
        dot.classList.add('ann-signal-dot');
        signalContainer.appendChild(dot);

        const delay = idx * 60;
        const startTime = performance.now() + delay;

        function step(now) {
            const elapsed = now - startTime;
            if (elapsed < 0) { requestAnimationFrame(step); return; }
            const t = Math.min(elapsed / duration, 1);
            const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // ease in-out
            dot.setAttribute('cx', x1 + (x2 - x1) * eased);
            dot.setAttribute('cy', y1 + (y2 - y1) * eased);
            if (t < 1) {
                requestAnimationFrame(step);
            } else {
                dot.remove();
                resolve();
            }
        }
        requestAnimationFrame(step);
    }));

    await Promise.all(promises);
}
