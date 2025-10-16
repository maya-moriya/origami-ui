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
        this.selectedCrease = null; // The selected crease edge [v1, v2]
        this.foldOptions = null; // Fold options from server
        this.isDraggingCrease = false;
        this.mousePos = { x: 0, y: 0 };
        this.previewSide = null; // 1 or -1 for fold preview
        
        // Visibility options
        this.showVertexIndexes = true;
        this.showHiddenEdges = true;
        
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
        
        // Calculate static scale using 50% of screen size
        this.calculateStaticScale(width, height);
        
        // Scale vertex radius based on the scale factor
        this.vertexRadius = Math.max(8, Math.min(16, this.scale * 0.15));
    }
    
    calculateStaticScale(canvasWidth, canvasHeight) {
        // Static centering - use 50% of screen size for a fixed coordinate space
        const minDimension = Math.min(canvasWidth, canvasHeight);
        const useableSize = minDimension * 0.5; // 50% of screen size
        this.scale = useableSize / 10; // 10 units range (0 to 10)
        
        // Center the coordinate system around (5, 5) which is middle of 0-10 range
        this.offsetX = canvasWidth / 2 - 5 * this.scale;
        this.offsetY = canvasHeight / 2 + 5 * this.scale; // Flip Y for canvas
    }
    
    recenterOrigami() {
        if (!this.origamiData || !this.origamiData.vertices) return;
        
        // Find bounding box of all vertices
        const vertices = Object.values(this.origamiData.vertices);
        if (vertices.length === 0) return;
        
        let minX = vertices[0][0], maxX = vertices[0][0];
        let minY = vertices[0][1], maxY = vertices[0][1];
        
        for (const vertex of vertices) {
            minX = Math.min(minX, vertex[0]);
            maxX = Math.max(maxX, vertex[0]);
            minY = Math.min(minY, vertex[1]);
            maxY = Math.max(maxY, vertex[1]);
        }
        
        // Calculate center and size of origami
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const width = maxX - minX;
        const height = maxY - minY;
        const maxDimension = Math.max(width, height);
        
        // Calculate new scale to fit origami with some padding
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        const minCanvasDimension = Math.min(canvasWidth, canvasHeight);
        const padding = 0.8; // Use 80% of canvas for origami, 20% for padding
        const targetSize = minCanvasDimension * padding;
        
        if (maxDimension > 0) {
            this.scale = targetSize / maxDimension;
        }
        
        // Center the origami in the canvas
        this.offsetX = canvasWidth / 2 - centerX * this.scale;
        this.offsetY = canvasHeight / 2 + centerY * this.scale; // Flip Y for canvas
        
        // Update vertex radius based on new scale
        this.vertexRadius = Math.max(8, Math.min(16, this.scale * 0.15));
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
        
        // Visibility checkboxes
        document.getElementById('show-vertex-indexes').addEventListener('change', (e) => {
            this.showVertexIndexes = e.target.checked;
            this.draw();
        });
        document.getElementById('show-hidden-edges').addEventListener('change', (e) => {
            this.showHiddenEdges = e.target.checked;
            this.draw();
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.draw();
        });
    }
    
    resetInteractionState() {
        this.hoveredVertex = null;
        this.hoveredEdge = null;
        this.isDragging = false;
        this.dragStartVertex = null;
        this.dragEndVertex = null;
        this.selectedCrease = null;
        this.foldOptions = null;
        this.isDraggingCrease = false;
        this.previewSide = null;
        document.getElementById('fold-selection').style.display = 'none';
        this.updateStatus('Ready - Drag between vertices to create a crease, or click on edges to split them');
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
            this.recenterOrigami();
            
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
        
        // Get face vertices in world coordinates
        const worldVertices = face.map(vid => {
            const vertex = this.origamiData.vertices[vid];
            return { vid, pos: [vertex[0], vertex[1]] };
        });
        
        // Check if we need to split this face for fold preview
        if (this.previewSide && this.foldOptions && this.selectedCrease) {
            const facesToHighlight = this.previewSide === 1 ? this.foldOptions.faces_positive : this.foldOptions.faces_negative;
            if (facesToHighlight.includes(parseInt(faceId))) {
                this.drawSplitFace(worldVertices, orientation, this.foldOptions.line, this.previewSide);
                return;
            }
        }
        
        // Draw normal face
        this.drawNormalFace(worldVertices, orientation, faceId);
    }
    
    drawNormalFace(worldVertices, orientation, faceId) {
        // Convert to canvas coordinates
        const points = worldVertices.map(v => this.worldToCanvas(v.pos[0], v.pos[1]));
        
        // Draw filled face
        this.ctx.beginPath();
        this.ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i].x, points[i].y);
        }
        this.ctx.closePath();
        
        // Normal colors - orientation 0 = white, 1 = lightblue
        this.ctx.globalAlpha = 1.0;
        const fillColor = orientation === 0 ? 'white' : 'lightblue';
        this.ctx.fillStyle = fillColor;
        
        // Debug logging for face rendering
        if (faceId) {
            console.log(`DEBUG: Rendering face ${faceId} with orientation ${orientation} as ${fillColor}`);
        }
        
        this.ctx.fill();
        
        // Draw face border
        this.ctx.strokeStyle = '#374151';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }
    
    drawSplitFace(worldVertices, orientation, line, previewSide) {
        // Split the face along the crease line
        const [A, B, C] = line;
        
        // Find vertices on the preview side and on the line
        const verticesOnSide = [];
        const verticesOnLine = [];
        const intersections = [];
        
        for (const vertex of worldVertices) {
            const d = A * vertex.pos[0] + B * vertex.pos[1] + C;
            if (Math.abs(d) < 1e-5) {
                verticesOnLine.push(vertex);
            } else if ((d > 0 && previewSide === 1) || (d < 0 && previewSide === -1)) {
                verticesOnSide.push(vertex);
            }
        }
        
        // Find intersections with face edges
        for (let i = 0; i < worldVertices.length; i++) {
            const v1 = worldVertices[i];
            const v2 = worldVertices[(i + 1) % worldVertices.length];
            
            const d1 = A * v1.pos[0] + B * v1.pos[1] + C;
            const d2 = A * v2.pos[0] + B * v2.pos[1] + C;
            
            // Check if edge crosses the line (different signs)
            if (d1 * d2 < 0) {
                const t = Math.abs(d1) / (Math.abs(d1) + Math.abs(d2));
                const intersection = [
                    v1.pos[0] + t * (v2.pos[0] - v1.pos[0]),
                    v1.pos[1] + t * (v2.pos[1] - v1.pos[1])
                ];
                intersections.push({ pos: intersection });
            }
        }
        
        // Combine vertices: on-side vertices + vertices on line + intersections
        const splitVertices = [...verticesOnSide, ...verticesOnLine, ...intersections];
        
        if (splitVertices.length < 3) {
            // If we can't form a proper polygon, draw the whole face normally
            this.drawNormalFace(worldVertices, orientation, 'split-fallback');
            return;
        }
        
        // Sort vertices to form a proper polygon (clockwise/counterclockwise)
        const sortedVertices = this.sortVerticesForPolygon(splitVertices);
        
        // Draw the split part with highlight
        const points = sortedVertices.map(v => this.worldToCanvas(v.pos[0], v.pos[1]));
        
        this.ctx.beginPath();
        this.ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i].x, points[i].y);
        }
        this.ctx.closePath();
        
        // Highlight color for fold preview
        this.ctx.globalAlpha = 1.0;
        this.ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
        this.ctx.fill();
        
        // Draw border
        this.ctx.strokeStyle = '#374151';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        
        // Also draw the remaining part with normal color
        this.drawRemainingFacePart(worldVertices, line, previewSide, orientation);
    }
    
    drawRemainingFacePart(worldVertices, line, previewSide, orientation) {
        const [A, B, C] = line;
        const oppositeSide = -previewSide;
        
        // Find vertices on the opposite side and on the line
        const verticesOnOppositeSide = [];
        const verticesOnLine = [];
        const intersections = [];
        
        for (const vertex of worldVertices) {
            const d = A * vertex.pos[0] + B * vertex.pos[1] + C;
            if (Math.abs(d) < 1e-5) {
                verticesOnLine.push(vertex);
            } else if ((d > 0 && oppositeSide === 1) || (d < 0 && oppositeSide === -1)) {
                verticesOnOppositeSide.push(vertex);
            }
        }
        
        // Find intersections with face edges (same logic as before)
        for (let i = 0; i < worldVertices.length; i++) {
            const v1 = worldVertices[i];
            const v2 = worldVertices[(i + 1) % worldVertices.length];
            
            const d1 = A * v1.pos[0] + B * v1.pos[1] + C;
            const d2 = A * v2.pos[0] + B * v2.pos[1] + C;
            
            if (d1 * d2 < 0) {
                const t = Math.abs(d1) / (Math.abs(d1) + Math.abs(d2));
                const intersection = [
                    v1.pos[0] + t * (v2.pos[0] - v1.pos[0]),
                    v1.pos[1] + t * (v2.pos[1] - v1.pos[1])
                ];
                intersections.push({ pos: intersection });
            }
        }
        
        const remainingVertices = [...verticesOnOppositeSide, ...verticesOnLine, ...intersections];
        
        if (remainingVertices.length < 3) return;
        
        const sortedVertices = this.sortVerticesForPolygon(remainingVertices);
        const points = sortedVertices.map(v => this.worldToCanvas(v.pos[0], v.pos[1]));
        
        this.ctx.beginPath();
        this.ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i].x, points[i].y);
        }
        this.ctx.closePath();
        
        // Normal color
        this.ctx.globalAlpha = 1.0;
        this.ctx.fillStyle = orientation === 0 ? 'white' : 'lightblue';
        this.ctx.fill();
        
        // Draw border
        this.ctx.strokeStyle = '#374151';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }
    
    sortVerticesForPolygon(vertices) {
        if (vertices.length < 3) return vertices;
        
        // Find centroid
        const centroid = [
            vertices.reduce((sum, v) => sum + v.pos[0], 0) / vertices.length,
            vertices.reduce((sum, v) => sum + v.pos[1], 0) / vertices.length
        ];
        
        // Sort by angle from centroid
        return vertices.sort((a, b) => {
            const angleA = Math.atan2(a.pos[1] - centroid[1], a.pos[0] - centroid[0]);
            const angleB = Math.atan2(b.pos[1] - centroid[1], b.pos[0] - centroid[0]);
            return angleA - angleB;
        });
    }
    
    drawAllEdges() {
        if (!this.showHiddenEdges) return;
        
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
        if (!this.origamiData || !this.origamiData.vertices) {
            return [];
        }
        
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
        if (!this.showVertexIndexes) return; // Don't draw vertices if hidden
        
        const pos = this.worldToCanvas(worldPos[0], worldPos[1]);
        const vertexId = parseInt(vid);
        
        // Determine fill color based on state
        const fillColor = this.getVertexFillColor(vertexId);
        
        // Draw vertex circle
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, this.vertexRadius, 0, 2 * Math.PI);
        this.ctx.fillStyle = fillColor;
        this.ctx.fill();
        
        // Enhanced stroke for different states
        if (this.hoveredVertex === vertexId) {
            this.ctx.strokeStyle = '#ef4444'; // Red stroke for hovered vertex
            this.ctx.lineWidth = 2.5;
        } else if (this.isDraggingCrease && this.dragEndVertex === vertexId) {
            this.ctx.strokeStyle = '#ef4444'; // Red stroke for drag target
            this.ctx.lineWidth = 2.5;
        } else if (this.selectedCrease && this.selectedCrease.includes(vertexId)) {
            this.ctx.strokeStyle = '#ef4444'; // Red stroke for selected crease vertices
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
        if (!this.showVertexIndexes) return; // Don't draw vertices if hidden
        
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
            
            // Enhanced stroke for different states
            if (this.hoveredVertex === vertexId) {
                this.ctx.strokeStyle = '#ef4444'; // Red stroke for hovered vertex
                this.ctx.lineWidth = 2.5;
            } else if (this.isDraggingCrease && this.dragEndVertex === vertexId) {
                this.ctx.strokeStyle = '#ef4444'; // Red stroke for drag target
                this.ctx.lineWidth = 2.5;
            } else if (this.selectedCrease && this.selectedCrease.includes(vertexId)) {
                this.ctx.strokeStyle = '#ef4444'; // Red stroke for selected crease vertices
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
        
        // Hover states take HIGHEST priority
        if (this.hoveredVertex === vertexId) {
            fillColor = 'rgba(239, 68, 68, 0.4)'; // Light red for hovered vertex
        }
        // Hovered edge vertices (show which edge will be split)
        else if (this.hoveredEdge && (this.hoveredEdge.includes(vertexId))) {
            fillColor = 'rgba(239, 68, 68, 0.2)'; // Light red for hovered edge vertices
        }
        // Drag states for crease creation
        else if (this.dragStartVertex === vertexId && this.isDraggingCrease) {
            fillColor = 'rgba(239, 68, 68, 0.6)'; // Darker red for drag start
        }
        else if (this.dragEndVertex === vertexId && this.isDraggingCrease) {
            fillColor = 'rgba(239, 68, 68, 0.6)'; // Darker red for drag end
        }
        // Selected crease vertices
        else if (this.selectedCrease && (this.selectedCrease.includes(vertexId))) {
            fillColor = '#ef4444'; // Solid red for selected crease vertices
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
            
            // Draw vertex ID (conditionally)
            if (this.showVertexIndexes) {
                this.ctx.globalAlpha = 1;
                this.ctx.fillStyle = '#374151';
                this.ctx.font = '500 11px -apple-system, BlinkMacSystemFont, sans-serif';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(this.getNextAvailableVertexId().toString(), previewX, previewY);
            }
            
            // Draw ratio label if it's a special ratio
            if (specialRatio) {
                this.ctx.fillStyle = '#3b82f6';
                this.ctx.font = '500 10px -apple-system, BlinkMacSystemFont, sans-serif';
                this.ctx.fillText(specialRatio.label, previewX, previewY - this.vertexRadius - 8);
            }
            
            this.ctx.restore();
        }
        
        // Draw selected crease line
        if (this.selectedCrease) {
            const v1 = this.origamiData.vertices[this.selectedCrease[0]];
            const v2 = this.origamiData.vertices[this.selectedCrease[1]];
            
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
        
        // Draw dragging line for crease creation
        if (this.isDraggingCrease && this.dragStart) {
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
        let bestMatch = null;
        let bestDistance = Infinity;
        let highestLayer = -1;
        
        for (const edge of this.origamiData.all_edges) {
            const v1 = this.origamiData.vertices[edge[0]];
            const v2 = this.origamiData.vertices[edge[1]];
            
            const p1 = this.worldToCanvas(v1[0], v1[1]);
            const p2 = this.worldToCanvas(v2[0], v2[1]);
            
            const result = this.distanceAndRatioToLineSegment(pos, p1, p2);
            if (result.distance <= threshold) {
                // Find the highest layer that contains this edge
                const edgeLayer = this.getHighestLayerForEdge(edge);
                
                // Prioritize by layer first, then by distance if same layer
                if (edgeLayer > highestLayer || 
                    (edgeLayer === highestLayer && result.distance < bestDistance)) {
                    bestMatch = { edge, ratio: result.ratio };
                    bestDistance = result.distance;
                    highestLayer = edgeLayer;
                }
            }
        }
        return bestMatch;
    }
    
    getHighestLayerForEdge(edge) {
        let highestLayer = 0;
        
        // Find all faces that contain this edge
        for (const [faceId, face] of Object.entries(this.origamiData.faces)) {
            if (face.includes(edge[0]) && face.includes(edge[1])) {
                // Find which layer this face is in
                for (const [layerNum, facesInLayer] of Object.entries(this.origamiData.layers)) {
                    if (facesInLayer.includes(parseInt(faceId))) {
                        highestLayer = Math.max(highestLayer, parseInt(layerNum));
                        break;
                    }
                }
            }
        }
        return highestLayer;
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
        if (this.isDraggingCrease) return;
        
        const pos = this.getMousePos(e);
        
        if (this.selectedCrease && this.foldOptions) {
            // We have a selected crease, perform fold based on mouse position
            if (this.previewSide) {
                this.performFold(this.selectedCrease, this.previewSide);
            }
        } else if (this.hoveredEdge && this.showVertexIndexes) {
            // Only allow edge splitting (vertex addition) when vertices are visible
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
    
    handleMouseDown(e) {
        const pos = this.getMousePos(e);
        
        // Only allow vertex interactions if vertices are visible
        if (this.showVertexIndexes) {
            const vertex = this.findVertexAt(pos);
            
            if (vertex) {
                // Start dragging to create a crease
                this.isDraggingCrease = true;
                this.dragStart = this.worldToCanvas(...this.origamiData.vertices[vertex]);
                this.dragStartVertex = vertex;
                this.canvas.style.cursor = 'grabbing';
                this.updateStatus('Drag to another vertex to create a crease line');
            }
        }
    }
    
    handleMouseMove(e) {
        this.mousePos = this.getMousePos(e);
        
        if (this.isDraggingCrease) {
            // Update drag end vertex only if vertices are visible
            this.dragEndVertex = this.showVertexIndexes ? this.findVertexAt(this.mousePos) : null;
            this.draw();
        } else {
            // Update hover states and fold preview
            this.updateHoverStates();
            
            if (this.selectedCrease && this.foldOptions) {
                this.updateFoldPreview();
            }
            
            // Update cursor based on what's under the mouse
            if (this.selectedCrease && this.previewSide) {
                this.canvas.style.cursor = 'pointer';
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
        if (this.isDraggingCrease) {
            const pos = this.getMousePos(e);
            // Only find vertex if vertices are visible
            const endVertex = this.showVertexIndexes ? this.findVertexAt(pos) : null;
            
            if (endVertex && endVertex !== this.dragStartVertex) {
                // Valid crease selection
                this.selectedCrease = [this.dragStartVertex, endVertex].sort((a, b) => a - b);
                this.loadFoldOptions();
                
                document.getElementById('selected-edge').textContent = `${this.selectedCrease[0]} - ${this.selectedCrease[1]}`;
                document.getElementById('fold-selection').style.display = 'block';
                this.updateStatus('Move mouse over faces to preview fold, then click to fold');
            } else {
                this.updateStatus('Drag between two different vertices to define crease line');
            }
            
            this.dragStartVertex = null;
            this.dragEndVertex = null;
        }
        
        this.isDraggingCrease = false;
        this.dragStart = null;
        this.canvas.style.cursor = 'crosshair';
        this.draw();
    }
    
    updateHoverStates() {
        if (!this.origamiData) {
            return;
        }
        
        this.hoveredVertex = null;
        this.hoveredEdge = null;
        this.hoveredEdgeRatio = 0.5;
        
        // Only check for vertex hover if vertices are visible
        if (this.showVertexIndexes) {
            this.hoveredVertex = this.findVertexAt(this.mousePos);
        }
        
        // Check for edges only if not hovering a vertex and not in fold mode and vertices are visible
        if (!this.hoveredVertex && !this.isDraggingCrease && !this.selectedCrease && this.showVertexIndexes) {
            const edgeResult = this.findEdgeAt(this.mousePos);
            if (edgeResult) {
                this.hoveredEdge = edgeResult.edge;
                this.hoveredEdgeRatio = edgeResult.ratio;
            }
        }
    }
    
    async loadFoldOptions() {
        try {
            const response = await fetch('/api/fold_options', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    edge: this.selectedCrease
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.foldOptions = result;
                console.log(`DEBUG: Loaded fold options:`, this.foldOptions);
            } else {
                this.updateStatus(`Error loading fold options: ${result.error}`);
            }
        } catch (error) {
            console.error('Fold options error:', error);
            this.updateStatus('Network error loading fold options');
        }
    }
    
    updateFoldPreview() {
        if (!this.foldOptions || !this.selectedCrease) {
            this.previewSide = null;
            return;
        }
        
        // Determine which side of the crease line the mouse is on
        const worldPos = this.canvasToWorld(this.mousePos.x, this.mousePos.y);
        const [A, B, C] = this.foldOptions.line;
        
        // Calculate which side of the line the mouse is on
        const d = A * worldPos.x + B * worldPos.y + C;
        const side = d > 1e-5 ? 1 : (d < -1e-5 ? -1 : 0);
        
        console.log(`DEBUG: Mouse at world pos (${worldPos.x.toFixed(2)}, ${worldPos.y.toFixed(2)})`);
        console.log(`DEBUG: Line equation: [${A.toFixed(4)}, ${B.toFixed(4)}, ${C.toFixed(4)}]`);
        console.log(`DEBUG: Distance d = ${d.toFixed(6)}, side = ${side}`);
        console.log(`DEBUG: Faces positive: [${this.foldOptions.faces_positive}], negative: [${this.foldOptions.faces_negative}]`);
        
        // Check if mouse is inside any face to set preview
        if (side !== 0 && this.isPointInAnyFace(worldPos)) {
            this.previewSide = side;
            console.log(`DEBUG: Preview side set to: ${this.previewSide}`);
        } else {
            this.previewSide = null;
            console.log(`DEBUG: Preview side cleared (side=${side}, inFace=${this.isPointInAnyFace(worldPos)})`);
        }
    }
    
    isPointInAnyFace(worldPos) {
        for (const [faceId, face] of Object.entries(this.origamiData.faces)) {
            if (this.isPointInFace(worldPos, face)) {
                return true;
            }
        }
        return false;
    }
    
    isPointInFace(point, face) {
        // Simple point-in-polygon test using ray casting
        const vertices = face.map(vid => this.origamiData.vertices[vid]);
        let inside = false;
        
        for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
            const xi = vertices[i][0], yi = vertices[i][1];
            const xj = vertices[j][0], yj = vertices[j][1];
            
            if (((yi > point.y) !== (yj > point.y)) &&
                (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
        return inside;
    }
    
    async performFold(edge, side) {
        try {
            console.log(`DEBUG: Performing fold with edge [${edge}] and side ${side}`);
            this.updateStatus('Folding...');
            this.canvas.style.cursor = 'wait';
            this.canvas.style.pointerEvents = 'none';
            
            const response = await fetch('/api/fold', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    edge: edge,
                    side: side
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.origamiData = result.state;
                console.log('DEBUG: Updated origami state after fold:', this.origamiData);
                console.log('DEBUG: Face orientations:', this.origamiData.faces_orientations);
                console.log('DEBUG: Faces:', this.origamiData.faces);
                console.log('DEBUG: Layers:', this.origamiData.layers);
                
                // Debug face positions
                console.log('DEBUG: Face positions:');
                for (const [faceId, face] of Object.entries(this.origamiData.faces)) {
                    const vertices = face.map(vid => {
                        const v = this.origamiData.vertices[vid];
                        return `${vid}:(${v[0].toFixed(1)},${v[1].toFixed(1)})`;
                    }).join(', ');
                    console.log(`  Face ${faceId}: [${vertices}] orientation:${this.origamiData.faces_orientations[faceId]}`);
                }
                
                // Debug layer order
                console.log('DEBUG: Layer rendering order (bottom to top):');
                const sortedLayerKeys = Object.keys(this.origamiData.layers).sort((a, b) => parseInt(a) - parseInt(b));
                for (const layerKey of sortedLayerKeys) {
                    const facesInLayer = this.origamiData.layers[layerKey];
                    console.log(`  Layer ${layerKey}: faces [${facesInLayer}]`);
                }
                this.updateVertexList();
                this.recenterOrigami();
                this.resetInteractionState();
                this.updateStatus('Fold completed successfully!');
            } else {
                this.updateStatus(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Fold error:', error);
            this.updateStatus('Network error during fold operation');
        } finally {
            this.canvas.style.pointerEvents = 'auto';
            this.canvas.style.cursor = 'crosshair';
        }
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
    
    
    async flip() {
        try {
            this.updateStatus('Flipping paper...');
            
            const response = await fetch('/api/flip', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.origamiData = result.state;
                this.recenterOrigami();
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
                this.recenterOrigami();
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
                this.recenterOrigami();
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