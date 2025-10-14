class OrigamiUI {
    constructor() {
        this.canvas = document.getElementById('origami-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.selectedEdge = null;
        this.isDragging = false;
        this.dragStart = null;
        this.origamiData = null;
        
        // Visual settings (will be updated by resizeCanvas)
        this.scale = 60; // Scale factor for display
        this.offsetX = 100; // Offset from canvas edge
        this.offsetY = 100;
        this.vertexRadius = 12;
        this.edgeThickness = 1;
        
        // Interaction state
        this.hoveredVertex = null;
        this.hoveredEdge = null;
        this.hoveredEdgeRatio = 0.5;
        this.dragStartVertex = null;
        this.dragEndVertex = null;
        this.foldingState = null; // null, 'line-selected', 'awaiting-vertex'
        this.flashingVertices = false;
        this.mousePos = { x: 0, y: 0 };
        this.hoveredDragTarget = null;
        this.lastDataHash = null; // Track when data changes
        
        // Special ratios to highlight
        this.specialRatios = [
            { value: 0.25, label: '0.25' },
            { value: 1/3, label: '1/3' },
            { value: 0.5, label: '0.5' },
            { value: 2/3, label: '2/3' },
            { value: 0.75, label: '0.75' }
        ];
        this.ratioThreshold = 0.05; // How close to a special ratio to snap
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        this.resizeCanvas();
        await this.loadState();
        this.draw();
    }
    
    resizeCanvas() {
        // Make canvas fill the entire left side of the screen
        const container = this.canvas.parentElement;
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';
        
        // Calculate scale based on origami bounds
        this.calculateOptimalScale(width, height);
        
        // Scale vertex radius based on the scale factor
        this.vertexRadius = Math.max(8, Math.min(16, this.scale * 0.15));
    }
    
    calculateOptimalScale(canvasWidth, canvasHeight) {
        if (!this.origamiData || !this.origamiData.faces) {
            // Fallback to default centering if no data
            const minDimension = Math.min(canvasWidth, canvasHeight);
            const margin = minDimension * 0.1;
            this.scale = (minDimension - 2 * margin) / 10;
            // Center around (5, 5) which is the middle of the default 10x10 space
            this.offsetX = canvasWidth / 2 - 5 * this.scale;
            this.offsetY = canvasHeight / 2 + 5 * this.scale; // Flip Y for canvas
            return;
        }
        
        // Calculate the center and bounds of all faces
        const faceCenter = this.calculateFacesCenter();
        const bounds = this.calculateOrigamiBounds();
        
        // Calculate optimal scale to fit the origami with margins
        const margin = Math.min(canvasWidth, canvasHeight) * 0.1;
        const availableWidth = canvasWidth - 2 * margin;
        const availableHeight = canvasHeight - 2 * margin;
        
        const origamiWidth = bounds.maxX - bounds.minX;
        const origamiHeight = bounds.maxY - bounds.minY;
        
        // Use the scale that fits both width and height
        const scaleX = origamiWidth > 0 ? availableWidth / origamiWidth : availableWidth / 10;
        const scaleY = origamiHeight > 0 ? availableHeight / origamiHeight : availableHeight / 10;
        this.scale = Math.min(scaleX, scaleY);
        
        // Center the origami based on its current face center
        this.offsetX = canvasWidth / 2 - faceCenter.x * this.scale;
        this.offsetY = canvasHeight / 2 + faceCenter.y * this.scale; // Flip Y for canvas
    }
    
    calculateFacesCenter() {
        if (!this.origamiData || !this.origamiData.faces) {
            return { x: 5, y: 5 }; // Default center
        }
        
        let totalX = 0;
        let totalY = 0;
        let faceCount = 0;
        
        // Calculate center of each face and average them
        for (const [faceId, face] of Object.entries(this.origamiData.faces)) {
            if (!face || face.length === 0) continue;
            
            // Calculate centroid of this face
            let faceX = 0;
            let faceY = 0;
            
            for (const vid of face) {
                const vertex = this.origamiData.vertices[vid];
                if (vertex) {
                    faceX += vertex[0];
                    faceY += vertex[1];
                }
            }
            
            faceX /= face.length;
            faceY /= face.length;
            
            totalX += faceX;
            totalY += faceY;
            faceCount++;
        }
        
        if (faceCount === 0) {
            return { x: 5, y: 5 }; // Default center
        }
        
        return {
            x: totalX / faceCount,
            y: totalY / faceCount
        };
    }
    
    calculateOrigamiBounds() {
        if (!this.origamiData || !this.origamiData.vertices) {
            return { minX: 0, maxX: 10, minY: 0, maxY: 10 }; // Default bounds
        }
        
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        
        for (const vertex of Object.values(this.origamiData.vertices)) {
            minX = Math.min(minX, vertex[0]);
            maxX = Math.max(maxX, vertex[0]);
            minY = Math.min(minY, vertex[1]);
            maxY = Math.max(maxY, vertex[1]);
        }
        
        // Add small padding to bounds
        const paddingX = (maxX - minX) * 0.05;
        const paddingY = (maxY - minY) * 0.05;
        
        return {
            minX: minX - paddingX,
            maxX: maxX + paddingX,
            minY: minY - paddingY,
            maxY: maxY + paddingY
        };
    }
    
    setupEventListeners() {
        // Canvas events
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        
        // Control buttons
        document.getElementById('flip-btn').addEventListener('click', () => this.flip());
        document.getElementById('undo-btn').addEventListener('click', () => this.undo());
        document.getElementById('reset-btn').addEventListener('click', () => this.reset());
        
        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.lastDataHash = null; // Force recalculation on resize
            this.draw();
        });
    }
    
    resetInteractionState() {
        this.selectedEdge = null;
        this.hoveredVertex = null;
        this.hoveredEdge = null;
        this.isDragging = false;
        this.dragStartVertex = null;
        this.dragEndVertex = null;
        this.foldingState = null;
        this.flashingVertices = false;
        document.getElementById('fold-selection').style.display = 'none';
        this.updateStatus('Ready - Click on edges to add vertices, or drag between vertices to fold');
        this.draw();
    }
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
    
    updateVertexList() {
        if (this.origamiData && this.origamiData.vertices) {
            const vertices = Object.keys(this.origamiData.vertices).sort((a, b) => parseInt(a) - parseInt(b));
            document.getElementById('vertex-list').textContent = vertices.join(', ');
        }
    }
    
    async loadState() {
        try {
            const response = await fetch('/api/state');
            this.origamiData = await response.json();
            this.updateVertexList();
            
        } catch (error) {
            console.error('Failed to load state:', error);
            this.updateStatus('Error loading origami state');
        }
    }
    
    getNextAvailableVertexId() {
        if (!this.origamiData || !this.origamiData.vertices) return 5;
        const existingIds = Object.keys(this.origamiData.vertices).map(id => parseInt(id));
        return Math.max(...existingIds) + 1;
    }
    
    // Coordinate transformation functions
    worldToCanvas(x, y) {
        return {
            x: x * this.scale + this.offsetX,
            y: -y * this.scale + this.offsetY  // Flip Y axis since canvas Y goes down
        };
    }
    
    canvasToWorld(canvasX, canvasY) {
        return {
            x: (canvasX - this.offsetX) / this.scale,
            y: -(canvasY - this.offsetY) / this.scale
        };
    }
    
    draw() {
        if (!this.origamiData) return;
        
        // Check if data has changed and recalculate positioning if needed
        const currentDataHash = this.getDataHash();
        if (currentDataHash !== this.lastDataHash) {
            this.calculateOptimalScale(this.canvas.width, this.canvas.height);
            this.lastDataHash = currentDataHash;
        }
        
        // Clear canvas with white background
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Reset any global alpha
        this.ctx.globalAlpha = 1.0;
        
        // Draw faces (sorted by layers, bottom first)
        const sortedFaces = this.getSortedFaces();
        
        for (const faceId of sortedFaces) {
            this.drawFace(faceId);
        }
        
        // Draw edges
        this.drawAllEdges();
        
        // Draw vertices
        this.drawVertices();
        
        // Draw selection overlay
        this.drawSelectionOverlay();
    }
    
    getDataHash() {
        if (!this.origamiData) return null;
        // Simple hash of face and vertex data to detect changes
        return JSON.stringify({
            faces: this.origamiData.faces,
            vertices: this.origamiData.vertices
        });
    }
    
    getSortedFaces() {
        const faces = [];
        const layers = this.origamiData.layers;
        
        // Sort layers by key (bottom to top)
        const sortedLayerKeys = Object.keys(layers).sort((a, b) => parseInt(a) - parseInt(b));
        
        for (const layerKey of sortedLayerKeys) {
            faces.push(...layers[layerKey]);
        }
        
        return faces;
    }
    
    drawFace(faceId) {
        const face = this.origamiData.faces[faceId];
        const orientation = this.origamiData.faces_orientations[faceId];
        
        if (!face || face.length < 3) return;
        
        
        // Get face vertices
        const points = face.map(vid => {
            const vertex = this.origamiData.vertices[vid];
            return this.worldToCanvas(vertex[0], vertex[1]);
        });
        
        // Draw filled face
        this.ctx.beginPath();
        this.ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i].x, points[i].y);
        }
        this.ctx.closePath();
        
        // Set fill color based on orientation - match source code exactly
        // Reset any transparency
        this.ctx.globalAlpha = 1.0;
        // Use matplotlib's 'white' and 'lightblue' colors - orientation 0 = white, 1 = lightblue
        this.ctx.fillStyle = orientation === 0 ? 'white' : 'lightblue';
        this.ctx.fill();
        
        // Draw face border with modern styling
        this.ctx.strokeStyle = '#374151';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }
    
    drawAllEdges() {
        const allEdges = this.origamiData.all_edges;
        
        this.ctx.strokeStyle = 'rgba(55, 65, 81, 0.2)';
        this.ctx.lineWidth = 0.5;
        
        for (const edge of allEdges) {
            const v1 = this.origamiData.vertices[edge[0]];
            const v2 = this.origamiData.vertices[edge[1]];
            
            const p1 = this.worldToCanvas(v1[0], v1[1]);
            const p2 = this.worldToCanvas(v2[0], v2[1]);
            
            this.ctx.beginPath();
            this.ctx.moveTo(p1.x, p1.y);
            this.ctx.lineTo(p2.x, p2.y);
            this.ctx.stroke();
        }
    }
    
    drawVertices() {
        // Group vertices by position (within tolerance)
        const positionGroups = this.groupVerticesByPosition();
        
        for (const group of positionGroups) {
            if (group.vertices.length === 1) {
                // Single vertex - draw normally
                this.drawSingleVertex(group.vertices[0], group.position);
            } else {
                // Multiple vertices at same position - draw as stacked circles
                this.drawGroupedVertices(group.vertices, group.position);
            }
        }
    }
    
    groupVerticesByPosition() {
        const tolerance = 1e-6; // Small tolerance for floating-point precision
        const groups = [];
        const processed = new Set();
        
        for (const [vid, vertex] of Object.entries(this.origamiData.vertices)) {
            if (processed.has(vid)) continue;
            
            const group = {
                position: [vertex[0], vertex[1]],
                vertices: [vid]
            };
            
            processed.add(vid);
            
            // Find other vertices at the same position
            for (const [otherId, otherVertex] of Object.entries(this.origamiData.vertices)) {
                if (processed.has(otherId)) continue;
                
                const dx = Math.abs(vertex[0] - otherVertex[0]);
                const dy = Math.abs(vertex[1] - otherVertex[1]);
                
                if (dx < tolerance && dy < tolerance) {
                    group.vertices.push(otherId);
                    processed.add(otherId);
                }
            }
            
            // Sort vertices in group by ID for consistent display
            group.vertices.sort((a, b) => parseInt(a) - parseInt(b));
            groups.push(group);
        }
        
        return groups;
    }
    
    drawSingleVertex(vid, worldPos) {
        const pos = this.worldToCanvas(worldPos[0], worldPos[1]);
        const vertexId = parseInt(vid);
        
        // Determine fill color based on state
        const fillColor = this.getVertexFillColor(vertexId);
        
        // Draw vertex circle
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, this.vertexRadius, 0, 2 * Math.PI);
        this.ctx.fillStyle = fillColor;
        this.ctx.fill();
        
        // Enhanced stroke for hovered vertices
        if (this.hoveredVertex === vertexId) {
            this.ctx.strokeStyle = '#ef4444'; // Red stroke for hovered vertex
            this.ctx.lineWidth = 2.5;
        } else {
            this.ctx.strokeStyle = '#374151'; // Normal stroke
            this.ctx.lineWidth = 1.5;
        }
        this.ctx.stroke();
        
        // Draw vertex label
        this.ctx.fillStyle = '#374151';
        this.ctx.font = '500 12px -apple-system, BlinkMacSystemFont, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(vid, pos.x, pos.y);
    }
    
    drawGroupedVertices(vids, worldPos) {
        const pos = this.worldToCanvas(worldPos[0], worldPos[1]);
        const numVertices = vids.length;
        
        // Increased spacing for better accessibility
        const spacing = this.vertexRadius * 1.2; // Increased from 0.3
        const radius = this.vertexRadius * 0.75; // Slightly larger circles
        
        // Calculate positions for stacked circles
        const startX = pos.x - (numVertices - 1) * spacing / 2;
        
        // Draw background highlight for the group
        const groupWidth = (numVertices - 1) * spacing + 2 * radius;
        this.ctx.beginPath();
        
        // Use roundRect if available, otherwise use regular rect
        if (this.ctx.roundRect) {
            this.ctx.roundRect(startX - radius - 2, pos.y - radius - 2, groupWidth + 4, 2 * radius + 4, 4);
        } else {
            this.ctx.rect(startX - radius - 2, pos.y - radius - 2, groupWidth + 4, 2 * radius + 4);
        }
        
        this.ctx.fillStyle = 'rgba(156, 163, 175, 0.1)';
        this.ctx.fill();
        this.ctx.strokeStyle = 'rgba(156, 163, 175, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        
        for (let i = 0; i < numVertices; i++) {
            const vid = vids[i];
            const vertexId = parseInt(vid);
            const circleX = startX + i * spacing;
            const circleY = pos.y;
            
            // Determine fill color based on state
            const fillColor = this.getVertexFillColor(vertexId);
            
            // Draw vertex circle with enhanced visibility
            this.ctx.beginPath();
            this.ctx.arc(circleX, circleY, radius, 0, 2 * Math.PI);
            this.ctx.fillStyle = fillColor;
            this.ctx.fill();
            
            // Enhanced stroke - stronger for hovered vertices
            if (this.hoveredVertex === vertexId) {
                this.ctx.strokeStyle = '#ef4444'; // Red stroke for hovered vertex
                this.ctx.lineWidth = 2.5;
            } else {
                this.ctx.strokeStyle = '#374151'; // Normal stroke
                this.ctx.lineWidth = 1.5;
            }
            this.ctx.stroke();
            
            // Draw vertex label with better contrast
            this.ctx.fillStyle = '#1f2937';
            this.ctx.font = '600 10px -apple-system, BlinkMacSystemFont, sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(vid, circleX, circleY);
        }
        
        // Add small indicator showing this is a grouped vertex set
        if (numVertices > 1) {
            const indicatorY = pos.y - radius - 8;
            this.ctx.fillStyle = '#6b7280';
            this.ctx.font = '500 8px -apple-system, BlinkMacSystemFont, sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`${numVertices} vertices`, pos.x, indicatorY);
        }
    }
    
    getVertexFillColor(vertexId) {
        let fillColor = '#ffffff';
        
        // Hover states take HIGHEST priority (always show hover feedback)
        if (this.hoveredVertex === vertexId) {
            // Stronger hover indication during flashing state
            if (this.foldingState === 'awaiting-vertex' && this.flashingVertices) {
                fillColor = 'rgba(239, 68, 68, 0.7)'; // Stronger red during flashing for clarity
            } else {
                fillColor = 'rgba(239, 68, 68, 0.4)'; // Normal red for other states
            }
        }
        // Drag states
        else if ((this.dragStartVertex === vertexId) || (this.dragEndVertex === vertexId)) {
            fillColor = '#ef4444'; // Red fill for selected drag vertices
        } else if (this.hoveredDragTarget === vertexId) {
            fillColor = 'rgba(239, 68, 68, 0.3)'; // Transparent red for drag target
        }
        // Flashing state (only if not being hovered)
        else if (this.foldingState === 'awaiting-vertex' && this.flashingVertices) {
            if (this.selectedEdge && (vertexId === this.selectedEdge[0] || vertexId === this.selectedEdge[1])) {
                fillColor = '#ffffff'; // Keep selected fold line vertices white
            } else {
                // Flashing yellow and white - but hover overrides this
                const time = Date.now();
                fillColor = Math.sin(time / 200) > 0 ? '#fef08a' : '#ffffff';
            }
        }
        // Default white
        else {
            fillColor = '#ffffff';
        }
        
        return fillColor;
    }
    
    drawSelectionOverlay() {
        // Draw hover preview for edge splitting
        if (this.hoveredEdge && !this.isDragging && this.foldingState !== 'awaiting-vertex') {
            const v1 = this.origamiData.vertices[this.hoveredEdge[0]];
            const v2 = this.origamiData.vertices[this.hoveredEdge[1]];
            
            const p1 = this.worldToCanvas(v1[0], v1[1]);
            const p2 = this.worldToCanvas(v2[0], v2[1]);
            
            // Check if we're close to a special ratio
            const ratio = this.hoveredEdgeRatio;
            let specialRatio = null;
            let finalRatio = ratio;
            
            for (const special of this.specialRatios) {
                if (Math.abs(ratio - special.value) < this.ratioThreshold) {
                    specialRatio = special;
                    finalRatio = special.value;
                    break;
                }
            }
            
            // Ensure we're not too close to the endpoints
            finalRatio = Math.max(0.05, Math.min(0.95, finalRatio));
            
            const previewX = p1.x + (p2.x - p1.x) * finalRatio;
            const previewY = p1.y + (p2.y - p1.y) * finalRatio;
            
            // Draw semi-transparent circle
            this.ctx.save();
            this.ctx.globalAlpha = 0.7;
            this.ctx.beginPath();
            this.ctx.arc(previewX, previewY, this.vertexRadius, 0, 2 * Math.PI);
            
            if (specialRatio) {
                this.ctx.fillStyle = 'rgba(59, 130, 246, 0.4)'; // Light blue for special ratios
            } else {
                this.ctx.fillStyle = 'rgba(156, 163, 175, 0.4)'; // Light gray for regular positions
            }
            
            this.ctx.fill();
            this.ctx.strokeStyle = specialRatio ? '#3b82f6' : '#9ca3af';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            // Draw vertex ID
            this.ctx.globalAlpha = 1;
            this.ctx.fillStyle = '#374151';
            this.ctx.font = '500 11px -apple-system, BlinkMacSystemFont, sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(this.getNextAvailableVertexId().toString(), previewX, previewY);
            
            // Draw ratio label if it's a special ratio
            if (specialRatio) {
                this.ctx.fillStyle = '#3b82f6';
                this.ctx.font = '500 10px -apple-system, BlinkMacSystemFont, sans-serif';
                this.ctx.fillText(specialRatio.label, previewX, previewY - this.vertexRadius - 8);
            }
            
            this.ctx.restore();
        }
        
        // Draw fold line
        if (this.selectedEdge) {
            const v1 = this.origamiData.vertices[this.selectedEdge[0]];
            const v2 = this.origamiData.vertices[this.selectedEdge[1]];
            
            const p1 = this.worldToCanvas(v1[0], v1[1]);
            const p2 = this.worldToCanvas(v2[0], v2[1]);
            
            this.ctx.strokeStyle = '#ef4444';
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([12, 6]);
            this.ctx.beginPath();
            this.ctx.moveTo(p1.x, p1.y);
            this.ctx.lineTo(p2.x, p2.y);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
        
        // Draw dragging line
        if (this.isDragging && this.dragStart) {
            this.ctx.strokeStyle = '#3b82f6';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([8, 4]);
            this.ctx.beginPath();
            this.ctx.moveTo(this.dragStart.x, this.dragStart.y);
            this.ctx.lineTo(this.mousePos.x, this.mousePos.y);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    findVertexAt(pos) {
        const positionGroups = this.groupVerticesByPosition();
        let bestMatch = null;
        let bestDistance = Infinity;
        
        for (const group of positionGroups) {
            if (group.vertices.length === 1) {
                // Single vertex - check normal radius
                const vPos = this.worldToCanvas(group.position[0], group.position[1]);
                const dx = pos.x - vPos.x;
                const dy = pos.y - vPos.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance <= this.vertexRadius && distance < bestDistance) {
                    bestMatch = parseInt(group.vertices[0]);
                    bestDistance = distance;
                }
            } else {
                // Grouped vertices - check each circle position individually
                const centerPos = this.worldToCanvas(group.position[0], group.position[1]);
                const numVertices = group.vertices.length;
                const spacing = this.vertexRadius * 0.6; // Match the drawing spacing
                const startX = centerPos.x - (numVertices - 1) * spacing / 2;
                const radius = this.vertexRadius * 0.75; // Match the drawing radius
                
                // Check each individual circle and find the closest one
                for (let i = 0; i < numVertices; i++) {
                    const circleX = startX + i * spacing;
                    const circleY = centerPos.y;
                    
                    const dx = pos.x - circleX;
                    const dy = pos.y - circleY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    // Only return the CLOSEST vertex within hit radius
                    if (distance <= radius && distance < bestDistance) {
                        bestMatch = parseInt(group.vertices[i]);
                        bestDistance = distance;
                    }
                }
            }
        }
        
        return bestMatch;
    }
    
    findEdgeAt(pos) {
        const threshold = 10; // pixels
        
        for (const edge of this.origamiData.all_edges) {
            const v1 = this.origamiData.vertices[edge[0]];
            const v2 = this.origamiData.vertices[edge[1]];
            
            const p1 = this.worldToCanvas(v1[0], v1[1]);
            const p2 = this.worldToCanvas(v2[0], v2[1]);
            
            const result = this.distanceAndRatioToLineSegment(pos, p1, p2);
            if (result.distance <= threshold) {
                return { edge, ratio: result.ratio };
            }
        }
        return null;
    }
    
    distanceToLineSegment(point, lineStart, lineEnd) {
        const result = this.distanceAndRatioToLineSegment(point, lineStart, lineEnd);
        return result.distance;
    }
    
    distanceAndRatioToLineSegment(point, lineStart, lineEnd) {
        const A = point.x - lineStart.x;
        const B = point.y - lineStart.y;
        const C = lineEnd.x - lineStart.x;
        const D = lineEnd.y - lineStart.y;
        
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        
        if (lenSq === 0) return { distance: Math.sqrt(A * A + B * B), ratio: 0 };
        
        let param = dot / lenSq;
        let distance;
        let ratio = Math.max(0, Math.min(1, param)); // Clamp ratio to [0, 1]
        
        if (param < 0) {
            distance = Math.sqrt(A * A + B * B);
        } else if (param > 1) {
            const dx = point.x - lineEnd.x;
            const dy = point.y - lineEnd.y;
            distance = Math.sqrt(dx * dx + dy * dy);
        } else {
            const projX = lineStart.x + param * C;
            const projY = lineStart.y + param * D;
            const dx = point.x - projX;
            const dy = point.y - projY;
            distance = Math.sqrt(dx * dx + dy * dy);
        }
        
        return { distance, ratio };
    }
    
    handleClick(e) {
        if (this.isDragging) return;
        
        const pos = this.getMousePos(e);
        
        if (this.foldingState === 'awaiting-vertex') {
            this.handleFoldVertexClick(pos);
        } else if (this.hoveredEdge) {
            this.handleEdgeClick(pos);
        }
    }
    
    async handleEdgeClick(pos) {
        if (this.hoveredEdge) {
            this.updateStatus('Adding vertex...');
            
            // Use the snapped ratio if we're close to a special ratio, otherwise use actual ratio
            let finalRatio = this.hoveredEdgeRatio;
            for (const special of this.specialRatios) {
                if (Math.abs(this.hoveredEdgeRatio - special.value) < this.ratioThreshold) {
                    finalRatio = special.value;
                    break;
                }
            }
            
            // Ensure ratio is within valid bounds
            finalRatio = Math.max(0.01, Math.min(0.99, finalRatio));
            
            await this.splitEdge(this.hoveredEdge, finalRatio);
        }
    }
    
    handleFoldVertexClick(pos) {
        const vertex = this.findVertexAt(pos);
        
        if (this.selectedEdge && vertex) {
            // Check if vertex is not on the selected edge
            if (vertex !== this.selectedEdge[0] && vertex !== this.selectedEdge[1]) {
                this.foldOnCrease(this.selectedEdge, vertex);
                this.resetInteractionState();
            } else {
                this.updateStatus('Select a vertex not on the fold line');
            }
        }
    }
    
    handleMouseDown(e) {
        const pos = this.getMousePos(e);
        const vertex = this.findVertexAt(pos);
        
        if (vertex && this.foldingState !== 'awaiting-vertex') {
            this.isDragging = true;
            this.dragStart = this.worldToCanvas(...this.origamiData.vertices[vertex]);
            this.dragStartVertex = vertex;
            this.canvas.style.cursor = 'grabbing';
        }
    }
    
    handleMouseMove(e) {
        this.mousePos = this.getMousePos(e);
        
        if (this.isDragging) {
            this.draw();
        } else {
            // Update hover states
            this.updateHoverStates();
            
            // Update cursor based on what's under the mouse
            if (this.foldingState === 'awaiting-vertex') {
                const vertex = this.findVertexAt(this.mousePos);
                this.canvas.style.cursor = vertex ? 'pointer' : 'crosshair';
            } else if (this.hoveredEdge) {
                this.canvas.style.cursor = 'pointer';
            } else if (this.hoveredVertex) {
                this.canvas.style.cursor = 'grab';
            } else {
                this.canvas.style.cursor = 'crosshair';
            }
            
            this.draw();
        }
    }
    
    handleMouseUp(e) {
        if (this.isDragging) {
            const pos = this.getMousePos(e);
            const endVertex = this.findVertexAt(pos);
            
            if (endVertex && endVertex !== this.dragStartVertex) {
                // Valid edge selection for folding
                this.selectedEdge = [this.dragStartVertex, endVertex].sort((a, b) => a - b);
                this.dragEndVertex = endVertex;
                this.foldingState = 'awaiting-vertex';
                this.flashingVertices = true;
                
                document.getElementById('selected-edge').textContent = `${this.selectedEdge[0]} - ${this.selectedEdge[1]}`;
                document.getElementById('fold-selection').style.display = 'block';
                this.updateStatus('Click on a vertex to choose which side to fold');
                
                // Start flashing animation
                this.startFlashingAnimation();
            } else {
                this.updateStatus('Drag between two different vertices to define fold line');
                this.dragStartVertex = null;
            }
        }
        
        this.isDragging = false;
        this.dragStart = null;
        this.canvas.style.cursor = 'crosshair';
        this.draw();
    }
    
    updateHoverStates() {
        this.hoveredVertex = null;
        this.hoveredEdge = null;
        this.hoveredEdgeRatio = 0.5;
        this.hoveredDragTarget = null;
        
        // Always check for vertex hover
        this.hoveredVertex = this.findVertexAt(this.mousePos);
        
        if (this.foldingState === 'awaiting-vertex') {
            // Only look for vertices when awaiting vertex selection - no edges
            return;
        }
        
        // If dragging, we still want to show hover on potential drag targets
        if (this.isDragging) {
            const dragTarget = this.findVertexAt(this.mousePos);
            if (dragTarget && dragTarget !== this.dragStartVertex) {
                this.hoveredDragTarget = dragTarget;
            }
        } 
        
        // Check for edges only if not hovering a vertex
        if (!this.hoveredVertex && !this.isDragging) {
            const edgeResult = this.findEdgeAt(this.mousePos);
            if (edgeResult) {
                this.hoveredEdge = edgeResult.edge;
                this.hoveredEdgeRatio = edgeResult.ratio;
            }
        }
    }
    
    startFlashingAnimation() {
        const flashInterval = setInterval(() => {
            if (this.foldingState !== 'awaiting-vertex') {
                clearInterval(flashInterval);
                return;
            }
            this.draw();
        }, 100);
    }
    
    async splitEdge(edge, ratio) {
        try {
            const response = await fetch('/api/split', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    edge: edge,
                    ratio: ratio
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.origamiData = result.state;
                this.updateVertexList();
                this.resetInteractionState();
                this.updateStatus(`Vertex added successfully! New vertex: ${result.new_vertex}`);
            } else {
                this.updateStatus(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Split error:', error);
            this.updateStatus('Network error during split operation');
        }
    }
    
    async foldOnCrease(edge, vertexToFold) {
        try {
            // Show immediate visual feedback
            this.updateStatus('Folding...');
            this.canvas.style.cursor = 'wait';
            
            // Disable further interactions
            this.canvas.style.pointerEvents = 'none';
            
            const response = await fetch('/api/fold', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    edge: edge,
                    vertex_to_fold: vertexToFold
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.origamiData = result.state;
                this.updateVertexList();
                this.updateStatus('Fold completed successfully!');
            } else {
                this.updateStatus(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Fold error:', error);
            this.updateStatus('Network error during fold operation');
        } finally {
            // Re-enable interactions
            this.canvas.style.pointerEvents = 'auto';
            this.canvas.style.cursor = 'crosshair';
        }
    }
    
    async flip() {
        try {
            this.updateStatus('Flipping paper...');
            
            const response = await fetch('/api/flip', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.origamiData = result.state;
                this.resetInteractionState();
                this.updateStatus('Paper flipped successfully!');
            } else {
                this.updateStatus(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Flip error:', error);
            this.updateStatus('Network error during flip operation');
        }
    }
    
    async undo() {
        try {
            this.updateStatus('Undoing last operation...');
            
            const response = await fetch('/api/undo', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.origamiData = result.state;
                this.updateVertexList();
                this.resetInteractionState();
                this.updateStatus('Operation undone successfully!');
            } else {
                this.updateStatus(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Undo error:', error);
            this.updateStatus('Network error during undo operation');
        }
    }
    
    async reset() {
        try {
            this.updateStatus('Resetting...');
            
            const response = await fetch('/api/reset', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.origamiData = result;
                this.updateVertexList();
                this.resetInteractionState();
                this.updateStatus('Reset to initial state');
            } else {
                this.updateStatus(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Reset error:', error);
            this.updateStatus('Network error during reset operation');
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new OrigamiUI();
});