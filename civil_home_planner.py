import streamlit as st
import requests
import json
from typing import Optional, Dict, List
from datetime import datetime
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon
import plotly.graph_objects as go
import plotly.express as px

class CivilHomePlanner:
    """Ollama-powered civil home planning assistant"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2:latest"
        
    def check_connection(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except requests.exceptions.RequestException:
            return []
    
    def get_design_suggestions(self, specifications: Dict) -> Optional[str]:
        """Get home design suggestions based on specifications"""
        try:
            prompt = f"""You are an expert civil engineer and home designer. 
            
Based on these specifications, provide detailed design suggestions:

Building Type: {specifications.get('building_type', 'Not specified')}
Total Area: {specifications.get('total_area', 'Not specified')} sq ft
Number of Floors: {specifications.get('num_floors', 'Not specified')}
Number of Bedrooms: {specifications.get('bedrooms', 'Not specified')}
Budget: {specifications.get('budget', 'Not specified')}
Climate/Location: {specifications.get('climate', 'Not specified')}
Special Requirements: {specifications.get('special_req', 'None')}

Please provide:
1. Room layout recommendations
2. Structural design suggestions
3. Material recommendations
4. Cost estimation breakdown
5. Timeline for construction
6. Compliance with building codes

Suggestions:"""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No suggestions generated.')
            else:
                st.error(f"Error from Ollama: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def get_material_recommendations(self, climate: str, budget_range: str) -> Optional[str]:
        """Get material recommendations based on climate and budget"""
        try:
            prompt = f"""You are a construction materials expert. 
            
Provide detailed material recommendations for a residential home project with these parameters:
- Climate: {climate}
- Budget Range: {budget_range}

Please recommend:
1. Foundation materials
2. Wall materials (walls, insulation, exterior finishes)
3. Roofing materials
4. Flooring materials
5. Window and door materials
6. Interior finishing materials
7. Plumbing materials
8. Electrical materials

Include cost estimates and pros/cons for each category.

Recommendations:"""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No recommendations generated.')
            return None
            
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def get_budget_estimation(self, area: float, quality: str, location: str) -> Optional[str]:
        """Get detailed cost estimation"""
        try:
            prompt = f"""You are a construction cost estimator. 

Provide a detailed budget breakdown for a residential construction project:
- Construction Area: {area} sq ft
- Quality Level: {quality} (Basic/Standard/Premium)
- Location Type: {location} (Urban/Suburban/Rural)

Provide:
1. Cost per square foot estimates
2. Detailed cost breakdown (Foundation, Walls, Roof, etc.)
3. Labor costs
4. Material costs
5. Contingency (10-15%)
6. Total estimated budget
7. Cost-saving tips
8. Premium options

Format with clear categories and estimated ranges.

Cost Estimation:"""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No estimation generated.')
            return None
            
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def get_compliance_guide(self, construction_type: str, location: str) -> Optional[str]:
        """Get building code and compliance guidelines"""
        try:
            prompt = f"""You are a building code compliance expert.

Provide comprehensive building code compliance guidelines for:
- Construction Type: {construction_type}
- Location/Region: {location}

Include:
1. Fire safety requirements
2. Structural requirements
3. Electrical code requirements
4. Plumbing code requirements
5. HVAC requirements
6. Accessibility requirements
7. Window and door specifications
8. Insulation requirements
9. Ventilation requirements
10. Permit and inspection checkpoints

Provide specific standards and recommendations.

Compliance Guide:"""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No guide generated.')
            return None
            
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def draw_detailed_floor_plan(self, bedrooms: int, bathrooms: int, area: float, style: str):
        """Draw detailed complete interior floor plan with furniture and fixtures"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        ax.set_xlim(0, 140)
        ax.set_ylim(0, 120)
        ax.set_aspect('equal')
        ax.set_facecolor('#fafafa')
        
        # Calculate dimensions based on area
        width = np.sqrt(area / 100) * 12
        height = area / width if width > 0 else 12
        
        # Scale to fit plot
        scale = 100 / max(width, height)
        width *= scale
        height *= scale
        
        start_x = 10
        start_y = 10
        
        # ============ OUTER WALLS ============
        outer_wall = Rectangle((start_x, start_y), width, height, 
                              linewidth=4, edgecolor='#2c3e50', facecolor='#ecf0f1', alpha=0.3)
        ax.add_patch(outer_wall)
        
        # ============ ROOMS DIMENSIONS ============
        main_height = height / 2.5
        hall_width = width / 2
        kitchen_width = width / 4
        
        # ============ LIVING ROOM / HALL ============
        living_rect = Rectangle((start_x + 1, start_y + height - main_height - 1), 
                               hall_width - 2, main_height - 2,
                               linewidth=2.5, edgecolor='#3498db', facecolor='#ebf5fb', alpha=0.6)
        ax.add_patch(living_rect)
        
        # Living room furniture
        # Sofa
        sofa = Rectangle((start_x + 3, start_y + height - main_height + 2), 8, 3, 
                        linewidth=2, edgecolor='#8b4513', facecolor='#d4a574')
        ax.add_patch(sofa)
        ax.text(start_x + 7, start_y + height - main_height + 3.5, 'SOFA', 
               ha='center', va='center', fontsize=8, weight='bold')
        
        # Coffee table
        coffee = Rectangle((start_x + 3, start_y + height - main_height + 6), 8, 2, 
                          linewidth=1.5, edgecolor='#654321', facecolor='#d2691e', alpha=0.7)
        ax.add_patch(coffee)
        ax.text(start_x + 7, start_y + height - main_height + 7, 'TABLE', 
               ha='center', va='center', fontsize=7, weight='bold', color='white')
        
        # TV Unit
        tv = Rectangle((start_x + 3, start_y + height - main_height + 9), 8, 2.5, 
                       linewidth=2, edgecolor='#2c3e50', facecolor='#34495e')
        ax.add_patch(tv)
        ax.text(start_x + 7, start_y + height - main_height + 10.2, 'TV UNIT', 
               ha='center', va='center', fontsize=7, weight='bold', color='white')
        
        # Dining table
        dining = Rectangle((start_x + 12, start_y + height - main_height + 3), 6, 5, 
                          linewidth=2, edgecolor='#8b4513', facecolor='#daa520', alpha=0.7)
        ax.add_patch(dining)
        ax.text(start_x + 15, start_y + height - main_height + 5.5, 'DINING', 
               ha='center', va='center', fontsize=8, weight='bold')
        
        # Chairs around dining
        for i, (x_off, y_off) in enumerate([(1, 0), (-1, 0), (0, 1.5), (0, -1.5)]):
            chair = Rectangle((start_x + 15 + x_off*1.5, start_y + height - main_height + 5.5 + y_off*1.5), 
                            1, 1, linewidth=1, edgecolor='brown', facecolor='#d4a574')
            ax.add_patch(chair)
        
        ax.text(start_x + hall_width/2, start_y + height - 2, 'LIVING ROOM / HALL', 
               ha='center', fontsize=11, weight='bold', style='italic', color='#2c3e50')
        
        # ============ KITCHEN ============
        kitchen_rect = Rectangle((start_x + hall_width + 1, start_y + height - main_height - 1), 
                                width - hall_width - 2, main_height - 2,
                                linewidth=2.5, edgecolor='#e74c3c', facecolor='#fadbd8', alpha=0.6)
        ax.add_patch(kitchen_rect)
        
        # Stove/Cooktop
        stove = Rectangle((start_x + hall_width + 3, start_y + height - main_height + 2), 3, 3, 
                         linewidth=2, edgecolor='#34495e', facecolor='#95a5a6')
        ax.add_patch(stove)
        # Burners
        for i in range(4):
            circle = Circle((start_x + hall_width + 3.75 + (i % 2)*1.5, start_y + height - main_height + 2.75 + (i // 2)*1.5), 
                          0.4, color='#7f8c8d', ec='black', linewidth=1)
            ax.add_patch(circle)
        ax.text(start_x + hall_width + 4.5, start_y + height - main_height + 5.5, 'STOVE', 
               ha='center', fontsize=7, weight='bold', color='white')
        
        # Refrigerator
        fridge = Rectangle((start_x + hall_width + 7, start_y + height - main_height + 2), 3, 3, 
                          linewidth=2, edgecolor='#2c3e50', facecolor='#bdc3c7')
        ax.add_patch(fridge)
        ax.text(start_x + hall_width + 8.5, start_y + height - main_height + 3.5, 'FRIDGE', 
               ha='center', fontsize=7, weight='bold')
        
        # Counter/Sink
        counter = Rectangle((start_x + hall_width + 11, start_y + height - main_height + 2), 4, 3, 
                           linewidth=2, edgecolor='#34495e', facecolor='#b0c4de')
        ax.add_patch(counter)
        # Sink
        sink = Rectangle((start_x + hall_width + 12, start_y + height - main_height + 2.5), 2, 1.5, 
                        linewidth=1, edgecolor='#2c3e50', facecolor='#e8e8e8')
        ax.add_patch(sink)
        ax.text(start_x + hall_width + 13, start_y + height - main_height + 3.2, 'SINK', 
               ha='center', fontsize=6, weight='bold')
        
        # Storage cabinets
        for j in range(2):
            cabinet = Rectangle((start_x + hall_width + 3 + j*4.5, start_y + height - main_height + 5.5), 
                               3, 2, linewidth=1.5, edgecolor='#654321', facecolor='#dcc9a5')
            ax.add_patch(cabinet)
        
        ax.text(start_x + hall_width + width/4, start_y + height - 2, 'KITCHEN', 
               ha='center', fontsize=11, weight='bold', style='italic', color='#2c3e50')
        
        # ============ BEDROOMS ============
        bedroom_height = (height - main_height) / bedrooms
        
        for bed_num in range(bedrooms):
            bed_y = start_y + (bedrooms - bed_num - 1) * bedroom_height
            
            # Bedroom area
            bedroom_rect = Rectangle((start_x + 1, bed_y + 1), hall_width - 2, bedroom_height - 2,
                                    linewidth=2.5, edgecolor='#9b59b6', facecolor='#f4ecf7', alpha=0.6)
            ax.add_patch(bedroom_rect)
            
            # Bed
            bed = Rectangle((start_x + 3, bed_y + 3), 6, 4, 
                           linewidth=2, edgecolor='#8b4513', facecolor='#f5deb3')
            ax.add_patch(bed)
            ax.text(start_x + 6, bed_y + 5, f'BED {bed_num + 1}', 
                   ha='center', va='center', fontsize=8, weight='bold')
            
            # Wardrobe
            wardrobe = Rectangle((start_x + 10, bed_y + 3), 3, 4, 
                                linewidth=2, edgecolor='#654321', facecolor='#d2691e', alpha=0.7)
            ax.add_patch(wardrobe)
            ax.text(start_x + 11.5, bed_y + 5, 'W', ha='center', va='center', 
                   fontsize=7, weight='bold', color='white')
            
            # Study table (in master bedroom)
            if bed_num == 0:
                study = Rectangle((start_x + 14, bed_y + 3), 3, 2, 
                                 linewidth=1.5, edgecolor='#8b4513', facecolor='#daa520', alpha=0.7)
                ax.add_patch(study)
                ax.text(start_x + 15.5, bed_y + 4, 'DESK', ha='center', fontsize=6, weight='bold')
            
            ax.text(start_x + hall_width/2, bed_y + bedroom_height - 1, f'BEDROOM {bed_num + 1}', 
                   ha='center', fontsize=9, weight='bold', style='italic', color='#2c3e50')
        
        # ============ BATHROOMS ============
        bath_y_start = start_y + height - main_height - (height - main_height) / (bathrooms + 1)
        bath_height = (height - main_height) / (bathrooms + 1.5)
        
        for bath_num in range(bathrooms):
            bath_y = bath_y_start - bath_num * bath_height
            
            # Bathroom area
            bathroom_rect = Rectangle((start_x + hall_width + 1, bath_y + 1), 
                                     width - hall_width - 2, bath_height - 2,
                                     linewidth=2.5, edgecolor='#16a085', facecolor='#d5f4e6', alpha=0.6)
            ax.add_patch(bathroom_rect)
            
            # Bathtub
            bathtub = Rectangle((start_x + hall_width + 3, bath_y + 2), 4, 2.5, 
                               linewidth=2, edgecolor='#2c3e50', facecolor='#ecf0f1')
            ax.add_patch(bathtub)
            ax.text(start_x + hall_width + 5, bath_y + 3.2, 'TUB', ha='center', fontsize=7, weight='bold')
            
            # Toilet
            toilet_bowl = Circle((start_x + hall_width + 9, bath_y + 3), 0.8, 
                               color='#ecf0f1', ec='#2c3e50', linewidth=2)
            ax.add_patch(toilet_bowl)
            ax.text(start_x + hall_width + 9, bath_y + 3, 'T', ha='center', va='center', 
                   fontsize=6, weight='bold')
            
            # Sink
            bath_sink = Rectangle((start_x + hall_width + 11, bath_y + 2), 3, 2, 
                                 linewidth=2, edgecolor='#2c3e50', facecolor='#bdc3c7')
            ax.add_patch(bath_sink)
            ax.text(start_x + hall_width + 12.5, bath_y + 3, 'SINK', ha='center', fontsize=7, weight='bold')
            
            ax.text(start_x + hall_width + (width - hall_width)/2, bath_y + bath_height - 0.5, 
                   f'BATHROOM {bath_num + 1}', ha='center', fontsize=8, weight='bold', 
                   style='italic', color='#2c3e50')
        
        # ============ DOORS & WINDOWS ============
        # Entry door
        door = Rectangle((start_x + hall_width - 1, start_y), 2, 0.5, 
                        linewidth=2, edgecolor='#8b4513', facecolor='#cd853f')
        ax.add_patch(door)
        ax.text(start_x + hall_width - 0.5, start_y - 1, '🚪 MAIN DOOR', ha='center', fontsize=8, weight='bold')
        
        # Windows (exterior walls)
        window_positions = [
            (start_x - 0.2, start_y + 15),  # Left wall
            (start_x - 0.2, start_y + 35),
            (start_x + width + 0.2, start_y + 15),  # Right wall
            (start_x + width + 0.2, start_y + 35),
            (start_x + 20, start_y - 0.2),  # Front wall
            (start_x + 40, start_y - 0.2),
        ]
        
        for wx, wy in window_positions[:min(4, 6)]:
            window = Rectangle((wx - 0.3, wy - 0.3), 0.6, 0.6, 
                             linewidth=2, edgecolor='#3498db', facecolor='#87ceeb', alpha=0.8)
            ax.add_patch(window)
        
        # ============ TITLE & MEASUREMENTS ============
        ax.text(70, 115, f'{style.upper()} HOME - COMPLETE FLOOR PLAN', 
               ha='center', fontsize=16, weight='bold', color='#2c3e50',
               bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2))
        
        # Measurements info
        info_text = f'Total Area: {area:.0f} sq ft | Width: {width:.1f}m | Length: {height:.1f}m\nBedrooms: {bedrooms} | Bathrooms: {bathrooms}'
        ax.text(70, 4, info_text, ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='#f9e79f', edgecolor='#2c3e50', linewidth=1.5))
        
        # Scale and legend
        ax.text(10, 4, '═════════════════════════════', ha='left', fontsize=9, family='monospace')
        ax.text(10, 1, 'LEGEND: Proper Scale ◯ Windows | 🚪 Doors | Measurements in meters', 
               ha='left', fontsize=9, weight='bold')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def draw_3d_house_view(self, bedrooms: int, style: str):
        """Create 3D house visualization"""
        # Create 3D plot data for house
        fig = go.Figure()
        
        # House dimensions
        length = 20
        width = 15
        height = 10
        
        # Draw house outline (3D representation)
        # Front face
        front_x = [0, length, length, 0, 0]
        front_y = [0, 0, 0, 0, 0]
        front_z = [0, 0, height, height, 0]
        
        fig.add_trace(go.Scatter3d(x=front_x, y=front_y, z=front_z, 
                                   mode='lines', name='Front Wall', 
                                   line=dict(color='black', width=2)))
        
        # Back face
        back_x = [0, length, length, 0, 0]
        back_y = [width, width, width, width, width]
        back_z = [0, 0, height, height, 0]
        
        fig.add_trace(go.Scatter3d(x=back_x, y=back_y, z=back_z,
                                   mode='lines', name='Back Wall',
                                   line=dict(color='black', width=2)))
        
        # Side connections
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, width], z=[0, 0],
                                   mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[length, length], y=[0, width], z=[0, 0],
                                   mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, width], z=[height, height],
                                   mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[length, length], y=[0, width], z=[height, height],
                                   mode='lines', line=dict(color='black', width=2), showlegend=False))
        
        # Roof (triangle)
        roof_peak_z = height + 5
        roof_x = [0, length, length/2, 0]
        roof_y = [0, 0, width/2, 0]
        roof_z = [height, height, roof_peak_z, height]
        
        fig.add_trace(go.Scatter3d(x=roof_x, y=roof_y, z=roof_z,
                                   mode='lines', name='Roof', 
                                   line=dict(color='red', width=3)))
        
        # Add windows (as points)
        window_x = [2, 8, 14, 2, 8, 14]
        window_y = [0, 0, 0, width, width, width]
        window_z = [height-4, height-4, height-4, height-4, height-4, height-4]
        
        fig.add_trace(go.Scatter3d(x=window_x, y=window_y, z=window_z,
                                   mode='markers', name='Windows',
                                   marker=dict(size=4, color='cyan', symbol='square')))
        
        # Add door
        fig.add_trace(go.Scatter3d(x=[length/2], y=[0], z=[height/2],
                                   mode='markers', name='Door',
                                   marker=dict(size=6, color='brown', symbol='diamond')))
        
        fig.update_layout(
            title=f'{style} House - 3D View ({bedrooms} Bedrooms)',
            scene=dict(
                xaxis_title='Length (m)',
                yaxis_title='Width (m)',
                zaxis_title='Height (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=900,
            height=700,
            showlegend=True
        )
        
        return fig
    
    def generate_3d_visualization_guide(self, style: str, rooms: str, materials: str) -> Optional[str]:
        """Generate SketchUp modeling and 3D visualization guide"""
        try:
            prompt = f"""You are an expert in SketchUp modeling and 3D architectural visualization.

Provide a comprehensive guide for creating a 3D model in SketchUp with these specifications:
- Architectural Style: {style}
- Rooms/Spaces: {rooms}
- Primary Materials: {materials}

Include:
1. SketchUp Setup & Best Practices
2. Step-by-step modeling instructions for:
   - Foundation and floor planning
   - Wall construction
   - Roof design
   - Doors and windows placement
3. Material and texture application
4. Lighting setup for realistic rendering
5. Camera angles for presentations
6. Adding details (fixtures, furniture, landscaping)
7. Rendering settings for high-quality output
8. Export options for presentations
9. Common mistakes to avoid
10. Tips for realistic visualization

Make it practical and actionable for someone using SketchUp.

3D Visualization Guide:"""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No guide generated.')
            return None
            
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def draw_interior_design_complete(self, bedrooms: int, bathrooms: int, area: float, style: str):
        """Draw complete interior design with 4 detailed room views"""
        fig = plt.figure(figsize=(20, 16))
        
        # Main grid for 4 subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # ==================== 1. LIVING ROOM HALL ====================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_aspect('equal')
        ax1.set_facecolor('#fef5e7')
        
        # Hall floor
        hall_floor = Rectangle((5, 5), 90, 70, linewidth=3, edgecolor='#2c3e50', facecolor='#f0e68c', alpha=0.3)
        ax1.add_patch(hall_floor)
        
        # Walls
        ax1.plot([5, 5, 95, 95, 5], [5, 75, 75, 5, 5], 'k-', linewidth=4)
        
        # Sofa
        sofa = Rectangle((10, 50), 20, 12, linewidth=2.5, edgecolor='#8b4513', facecolor='#d4a574', alpha=0.8)
        ax1.add_patch(sofa)
        ax1.text(20, 56, 'SOFA', ha='center', va='center', fontsize=11, weight='bold', color='#2c3e50')
        
        # Coffee table
        coffee = Rectangle((35, 48), 15, 10, linewidth=2, edgecolor='#654321', facecolor='#cd853f', alpha=0.7)
        ax1.add_patch(coffee)
        ax1.text(42.5, 53, 'COFFEE\nTABLE', ha='center', va='center', fontsize=9, weight='bold', color='white')
        
        # TV Unit
        tv_unit = Rectangle((60, 45), 25, 15, linewidth=2.5, edgecolor='#2c3e50', facecolor='#34495e', alpha=0.8)
        ax1.add_patch(tv_unit)
        # TV screen
        tv_screen = Rectangle((63, 50), 18, 8, linewidth=1, edgecolor='black', facecolor='#1c1c1c', alpha=0.9)
        ax1.add_patch(tv_screen)
        ax1.text(72, 54, '📺 TV', ha='center', va='center', fontsize=10, weight='bold', color='white')
        
        # Dining table
        dining = Rectangle((15, 15), 25, 20, linewidth=2.5, edgecolor='#8b4513', facecolor='#daa520', alpha=0.7)
        ax1.add_patch(dining)
        ax1.text(27.5, 25, 'DINING\nTABLE', ha='center', va='center', fontsize=10, weight='bold', color='#2c3e50')
        
        # Dining chairs (4 around table)
        for cx, cy in [(10, 15), (40, 15), (10, 35), (40, 35)]:
            chair = Rectangle((cx, cy), 4, 4, linewidth=1.5, edgecolor='#654321', facecolor='#d4a574')
            ax1.add_patch(chair)
            ax1.text(cx+2, cy+2, '🪑', ha='center', va='center', fontsize=8)
        
        # Wall decorations
        ax1.text(8, 40, '🖼️', fontsize=16)
        ax1.text(92, 60, '🖼️', fontsize=16)
        ax1.text(8, 10, '🪴', fontsize=14)
        
        # Windows
        ax1.add_patch(Rectangle((25, 75.5), 15, 2, linewidth=2, edgecolor='#3498db', facecolor='#87ceeb', alpha=0.8))
        ax1.add_patch(Rectangle((60, 75.5), 15, 2, linewidth=2, edgecolor='#3498db', facecolor='#87ceeb', alpha=0.8))
        
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.text(50, 95, '🏠 LIVING ROOM / HALL', ha='center', fontsize=13, weight='bold', style='italic')
        ax1.text(50, 0, 'Main entertaining space with comfortable seating & dining', ha='center', fontsize=9, style='italic')
        
        # ==================== 2. KITCHEN ====================
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.set_aspect('equal')
        ax2.set_facecolor('#fef5e7')
        
        # Kitchen floor
        kitchen_floor = Rectangle((5, 5), 90, 70, linewidth=3, edgecolor='#2c3e50', facecolor='#fff8dc', alpha=0.4)
        ax2.add_patch(kitchen_floor)
        
        # Walls
        ax2.plot([5, 5, 95, 95, 5], [5, 75, 75, 5, 5], 'k-', linewidth=4)
        
        # Countertop (L-shaped)
        counter_h = Rectangle((10, 10), 35, 8, linewidth=2, edgecolor='#34495e', facecolor='#95a5a6', alpha=0.7)
        ax2.add_patch(counter_h)
        counter_v = Rectangle((45, 10), 8, 30, linewidth=2, edgecolor='#34495e', facecolor='#95a5a6', alpha=0.7)
        ax2.add_patch(counter_v)
        
        # Stove
        stove = Rectangle((12, 20), 10, 10, linewidth=2.5, edgecolor='#2c3e50', facecolor='#7f8c8d', alpha=0.8)
        ax2.add_patch(stove)
        for i, (ox, oy) in enumerate([(2, 2), (7, 2), (2, 7), (7, 7)]):
            circle = Circle((14+ox, 25+oy), 1.2, color='#34495e', ec='black', linewidth=1.5)
            ax2.add_patch(circle)
        ax2.text(17, 15, 'STOVE', ha='center', fontsize=9, weight='bold', color='white')
        
        # Refrigerator
        fridge = Rectangle((30, 18), 12, 14, linewidth=2.5, edgecolor='#34495e', facecolor='#b0c4de', alpha=0.8)
        ax2.add_patch(fridge)
        ax2.text(36, 25, 'FRIDGE', ha='center', fontsize=9, weight='bold', color='#2c3e50')
        
        # Sink
        sink = Rectangle((12, 30), 12, 8, linewidth=2, edgecolor='#2c3e50', facecolor='#b0e0e6', alpha=0.8)
        ax2.add_patch(sink)
        ax2.text(18, 34, 'SINK', ha='center', fontsize=8, weight='bold', color='#2c3e50')
        
        # Microwave
        microwave = Rectangle((28, 32), 8, 8, linewidth=2, edgecolor='#34495e', facecolor='#d3d3d3', alpha=0.7)
        ax2.add_patch(microwave)
        ax2.text(32, 36, '📶', ha='center', fontsize=10)
        
        # Island/Prep table
        island = Rectangle((55, 25), 25, 18, linewidth=2.5, edgecolor='#8b4513', facecolor='#daa520', alpha=0.6)
        ax2.add_patch(island)
        ax2.text(67.5, 34, 'PREP/\nDINING', ha='center', va='center', fontsize=9, weight='bold', color='#2c3e50')
        
        # Stools
        for sx in [58, 63, 68, 73, 78]:
            stool = Rectangle((sx, 20), 3, 3, linewidth=1.5, edgecolor='#654321', facecolor='#d4a574')
            ax2.add_patch(stool)
        
        # Cabinet storage
        for cx in [10, 25, 40]:
            cabinet = Rectangle((cx, 40), 10, 15, linewidth=1.5, edgecolor='#654321', facecolor='#dcc9a5', alpha=0.7)
            ax2.add_patch(cabinet)
        
        # Window
        ax2.add_patch(Rectangle((70, 75.5), 20, 2, linewidth=2, edgecolor='#3498db', facecolor='#87ceeb', alpha=0.8))
        
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.text(50, 95, '🍳 KITCHEN', ha='center', fontsize=13, weight='bold', style='italic')
        ax2.text(50, 0, 'Well-equipped kitchen with stove, refrigerator, sink & storage', ha='center', fontsize=9, style='italic')
        
        # ==================== 3. MASTER BEDROOM ====================
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 100)
        ax3.set_aspect('equal')
        ax3.set_facecolor('#fef5e7')
        
        # Bedroom floor
        bed_floor = Rectangle((5, 5), 90, 70, linewidth=3, edgecolor='#2c3e50', facecolor='#f4ecf7', alpha=0.4)
        ax3.add_patch(bed_floor)
        
        # Walls
        ax3.plot([5, 5, 95, 95, 5], [5, 75, 75, 5, 5], 'k-', linewidth=4)
        
        # King bed
        bed = Rectangle((15, 35), 40, 30, linewidth=2.5, edgecolor='#8b4513', facecolor='#f5deb3', alpha=0.8)
        ax3.add_patch(bed)
        pillow = Rectangle((15, 60), 40, 5, linewidth=1, edgecolor='#daa520', facecolor='#fff8dc', alpha=0.9)
        ax3.add_patch(pillow)
        ax3.text(35, 50, '🛏️ BED', ha='center', va='center', fontsize=12, weight='bold', color='#2c3e50')
        
        # Bedside tables
        for bx in [10, 60]:
            bedside = Rectangle((bx, 45), 8, 10, linewidth=1.5, edgecolor='#654321', facecolor='#d2691e', alpha=0.7)
            ax3.add_patch(bedside)
            ax3.text(bx+4, 50, '💡', ha='center', fontsize=9)
        
        # Wardrobe
        wardrobe = Rectangle((62, 25), 18, 35, linewidth=2.5, edgecolor='#654321', facecolor='#d2691e', alpha=0.8)
        ax3.add_patch(wardrobe)
        ax3.text(71, 42.5, 'WARDROBE', ha='center', va='center', fontsize=9, weight='bold', color='white')
        ax3.plot([62, 80], [35, 35], 'k-', linewidth=1)
        ax3.plot([62, 80], [45, 45], 'k-', linewidth=1)
        
        # Study desk
        desk = Rectangle((10, 10), 20, 12, linewidth=2, edgecolor='#8b4513', facecolor='#daa520', alpha=0.7)
        ax3.add_patch(desk)
        ax3.text(20, 16, 'DESK', ha='center', fontsize=8, weight='bold', color='#2c3e50')
        
        # Study chair
        chair = Rectangle((32, 10), 6, 8, linewidth=1.5, edgecolor='#654321', facecolor='#d4a574')
        ax3.add_patch(chair)
        ax3.text(35, 14, '🪑', ha='center', fontsize=9)
        
        # Sofa/Lounge
        sofa_bed = Rectangle((45, 10), 15, 12, linewidth=2, edgecolor='#8b4513', facecolor='#d4a574', alpha=0.7)
        ax3.add_patch(sofa_bed)
        ax3.text(52.5, 16, 'LOUNGE', ha='center', fontsize=8, weight='bold', color='#2c3e50')
        
        # Windows (2 large windows)
        ax3.add_patch(Rectangle((25, 75.5), 18, 2, linewidth=2, edgecolor='#3498db', facecolor='#87ceeb', alpha=0.8))
        ax3.add_patch(Rectangle((60, 75.5), 18, 2, linewidth=2, edgecolor='#3498db', facecolor='#87ceeb', alpha=0.8))
        
        # Decorations
        ax3.text(8, 70, '🖼️', fontsize=14)
        ax3.text(90, 25, '🪴', fontsize=13)
        
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.text(50, 95, '🛏️ MASTER BEDROOM', ha='center', fontsize=13, weight='bold', style='italic')
        ax3.text(50, 0, 'Spacious bedroom with king bed, wardrobe, study area & seating', ha='center', fontsize=9, style='italic')
        
        # ==================== 4. BATHROOM ====================
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        ax4.set_aspect('equal')
        ax4.set_facecolor('#fef5e7')
        
        # Bathroom floor (tile pattern)
        bath_floor = Rectangle((5, 5), 90, 70, linewidth=3, edgecolor='#2c3e50', facecolor='#d5f4e6', alpha=0.4)
        ax4.add_patch(bath_floor)
        
        # Walls
        ax4.plot([5, 5, 95, 95, 5], [5, 75, 75, 5, 5], 'k-', linewidth=4)
        
        # Bathtub
        bathtub = Rectangle((10, 40), 30, 20, linewidth=2.5, edgecolor='#2c3e50', facecolor='#e8f8f5', alpha=0.9)
        ax4.add_patch(bathtub)
        ax4.text(11, 38, '🚿', fontsize=12)
        ax4.text(30, 60, 'BATHTUB', ha='center', fontsize=10, weight='bold', color='#2c3e50')
        
        # Shower area
        shower = Rectangle((50, 35), 20, 25, linewidth=2, edgecolor='#2c3e50', facecolor='#b0e0e6', alpha=0.6)
        ax4.add_patch(shower)
        ax4.text(60, 47.5, 'SHOWER\nCUBICLE', ha='center', fontsize=9, weight='bold', color='#2c3e50')
        ax4.text(60, 62, '🚿', fontsize=14)
        
        # Toilet
        toilet_circle = Circle((75, 50), 5, color='#ecf0f1', ec='#2c3e50', linewidth=2.5)
        ax4.add_patch(toilet_circle)
        ax4.text(75, 50, 'TOILET', ha='center', va='center', fontsize=8, weight='bold', color='#2c3e50')
        
        # Vanity/Sink
        vanity = Rectangle((10, 15), 35, 12, linewidth=2, edgecolor='#34495e', facecolor='#95a5a6', alpha=0.7)
        ax4.add_patch(vanity)
        ax4.add_patch(Rectangle((13, 18), 7, 7, linewidth=1.5, edgecolor='#2c3e50', facecolor='#b0e0e6', alpha=0.8))
        ax4.add_patch(Rectangle((28, 18), 7, 7, linewidth=1.5, edgecolor='#2c3e50', facecolor='#b0e0e6', alpha=0.8))
        ax4.text(17, 15, 'SINK', ha='center', fontsize=7, weight='bold')
        ax4.text(32, 15, 'SINK', ha='center', fontsize=7, weight='bold')
        ax4.text(30, 25, 'VANITY COUNTER', ha='center', fontsize=8, weight='bold', color='white')
        
        # Mirror
        ax4.add_patch(Rectangle((48, 18), 25, 12, linewidth=2, edgecolor='#7f8c8d', facecolor='#e8e8e8', alpha=0.7))
        ax4.text(60.5, 24, '🪞 MIRROR', ha='center', fontsize=9, weight='bold', color='#2c3e50')
        
        # Storage cabinet
        cabinet = Rectangle((75, 15), 15, 12, linewidth=1.5, edgecolor='#654321', facecolor='#d2691e', alpha=0.6)
        ax4.add_patch(cabinet)
        ax4.text(82.5, 21, 'STORAGE', ha='center', fontsize=7, weight='bold', color='white')
        
        # Window/Exhaust vent
        ax4.add_patch(Rectangle((40, 75.5), 20, 2, linewidth=2, edgecolor='#3498db', facecolor='#87ceeb', alpha=0.8))
        ax4.text(80, 10, '💨', fontsize=11)
        
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.text(50, 95, '🚿 BATHROOM', ha='center', fontsize=13, weight='bold', style='italic')
        ax4.text(50, 0, 'Modern bathroom with bathtub, shower, toilet, dual sinks & storage', ha='center', fontsize=9, style='italic')
        
        # ==================== MAIN TITLE ====================
        fig.suptitle(f'🏠 COMPLETE {style.upper()} HOME - INTERIOR DESIGN LAYOUT 🏠', 
                    fontsize=18, weight='bold', y=0.98, color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#f9e79f', edgecolor='#2c3e50', linewidth=2))
        
        # Overall info
        info_text = f'Total Area: {area:.0f} sq ft | Bedrooms: {bedrooms} | Bathrooms: {bathrooms}'
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=11, weight='bold', style='italic',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        return fig


def main():
    st.set_page_config(
        page_title="Civil Home Planner with Ollama",
        page_icon="🏠",
        layout="wide"
    )
    
    # Custom styling
    st.markdown("""
    <style>
    .main-header { font-size: 3em; color: #1f77b4; margin-bottom: 10px; }
    .subheader { font-size: 1.5em; color: #555; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏠 Civil Home Planner with Ollama")
    st.markdown("AI-powered assistance for residential home planning, design, and construction!")
    
    # Initialize planner
    planner = CivilHomePlanner()
    
    # Sidebar - Connection and Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Ollama host configuration
        ollama_host = st.text_input(
            "Ollama Host:",
            value="http://localhost:11434",
            help="Enter Ollama server URL"
        )
        
        if ollama_host != planner.base_url:
            planner = CivilHomePlanner(ollama_host)
        
        # Connection status
        if planner.check_connection():
            st.success("✅ Ollama Connected!")
            
            models = planner.get_available_models()
            if models:
                selected_model = st.selectbox("Select Model:", models)
                planner.model = selected_model
                st.info(f"📌 Using: {selected_model}")
            else:
                st.error("❌ No models available")
                return
        else:
            st.error("❌ Cannot connect to Ollama")
            st.info("Start Ollama: `ollama serve`")
            return
        
        st.divider()
        st.markdown("**📋 Navigation**")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎨 Design Suggestions",
        "🛠️ Materials",
        "💰 Budget",
        "📋 Compliance",
        "📐 Floor Plans",
        "🏗️ SketchUp Guide"
    ])
    
    # Tab 1: Design Suggestions
    with tab1:
        st.header("🎨 Design Suggestions")
        st.markdown("Get AI-powered design recommendations for your home project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            building_type = st.selectbox(
                "Building Type",
                ["Single Family Home", "Apartment", "Duplex", "Villa", "Cottage", "Modern House"]
            )
            total_area = st.number_input("Total Area (sq ft)", min_value=500, max_value=50000, value=2000)
            num_floors = st.slider("Number of Floors", 1, 5, 2)
        
        with col2:
            bedrooms = st.slider("Bedrooms", 1, 10, 3)
            budget = st.selectbox("Budget Level", ["Budget", "Standard", "Premium", "Luxury"])
            climate = st.selectbox(
                "Climate/Location",
                ["Tropical", "Temperate", "Cold", "Arid", "Humid"]
            )
        
        special_req = st.text_area(
            "Special Requirements (Optional)",
            placeholder="e.g., Open concept kitchen, Home office, Swimming pool..."
        )
        
        if st.button("🚀 Get Design Suggestions", type="primary", use_container_width=True):
            with st.spinner("Analyzing your requirements..."):
                specifications = {
                    "building_type": building_type,
                    "total_area": total_area,
                    "num_floors": num_floors,
                    "bedrooms": bedrooms,
                    "budget": budget,
                    "climate": climate,
                    "special_req": special_req
                }
                
                suggestions = planner.get_design_suggestions(specifications)
                
                if suggestions:
                    st.session_state['design_suggestions'] = suggestions
        
        if 'design_suggestions' in st.session_state:
            st.success("✅ Design Suggestions Generated!")
            st.markdown("---")
            st.markdown(st.session_state['design_suggestions'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "💾 Download Suggestions",
                    st.session_state['design_suggestions'],
                    file_name="design_suggestions.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # Tab 2: Material Recommendations
    with tab2:
        st.header("🛠️ Material Recommendations")
        st.markdown("Get expert material recommendations tailored to your project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            material_climate = st.selectbox(
                "Climate",
                ["Tropical", "Temperate", "Cold", "Arid", "Humid", "Monsoon"],
                key="material_climate"
            )
        
        with col2:
            material_budget = st.selectbox(
                "Budget Range",
                ["Budget (Basic)", "Moderate", "Mid-Range", "Premium", "Luxury"],
                key="material_budget"
            )
        
        if st.button("🔍 Get Material Recommendations", type="primary", use_container_width=True):
            with st.spinner("Researching materials..."):
                recommendations = planner.get_material_recommendations(
                    material_climate,
                    material_budget
                )
                
                if recommendations:
                    st.session_state['material_recs'] = recommendations
        
        if 'material_recs' in st.session_state:
            st.success("✅ Material Recommendations Generated!")
            st.markdown("---")
            st.markdown(st.session_state['material_recs'])
            
            st.download_button(
                "💾 Download Materials List",
                st.session_state['material_recs'],
                file_name="materials_recommendations.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Tab 3: Budget Estimation
    with tab3:
        st.header("💰 Budget Estimation")
        st.markdown("Get detailed cost breakdown for your home project")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget_area = st.number_input(
                "Construction Area (sq ft)",
                min_value=500,
                max_value=50000,
                value=2000,
                key="budget_area"
            )
        
        with col2:
            quality_level = st.selectbox(
                "Quality Level",
                ["Basic", "Standard", "Premium"],
                key="quality_level"
            )
        
        with col3:
            location_type = st.selectbox(
                "Location Type",
                ["Urban", "Suburban", "Rural"],
                key="location_type"
            )
        
        if st.button("🧮 Calculate Budget", type="primary", use_container_width=True):
            with st.spinner("Calculating costs..."):
                estimation = planner.get_budget_estimation(
                    budget_area,
                    quality_level,
                    location_type
                )
                
                if estimation:
                    st.session_state['budget_est'] = estimation
        
        if 'budget_est' in st.session_state:
            st.success("✅ Budget Estimation Generated!")
            st.markdown("---")
            st.markdown(st.session_state['budget_est'])
            
            st.download_button(
                "💾 Download Budget Report",
                st.session_state['budget_est'],
                file_name="budget_estimation.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Tab 4: Building Code Compliance
    with tab4:
        st.header("📋 Building Code & Compliance")
        st.markdown("Get comprehensive compliance guidelines for your project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            construction_type = st.selectbox(
                "Construction Type",
                ["Residential Single Family", "Residential Multi-Family", "Commercial", "Mixed Use"],
                key="construction_type"
            )
        
        with col2:
            compliance_location = st.text_input(
                "Location/Region",
                placeholder="e.g., California, Texas, New York, etc.",
                key="compliance_location"
            )
        
        if st.button("📚 Get Compliance Guide", type="primary", use_container_width=True):
            with st.spinner("Researching compliance requirements..."):
                compliance = planner.get_compliance_guide(
                    construction_type,
                    compliance_location
                )
                
                if compliance:
                    st.session_state['compliance_guide'] = compliance
        
        if 'compliance_guide' in st.session_state:
            st.success("✅ Compliance Guide Generated!")
            st.markdown("---")
            st.markdown(st.session_state['compliance_guide'])
            
            st.download_button(
                "💾 Download Compliance Guide",
                st.session_state['compliance_guide'],
                file_name="compliance_guide.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Tab 5: Floor Plans & Sketches
    with tab5:
        st.header("📐 Floor Plan Sketches")
        st.markdown("Generate detailed architectural floor plan drawings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fp_bedrooms = st.slider("Number of Bedrooms", 1, 8, 3, key="fp_bedrooms")
            fp_bathrooms = st.slider("Number of Bathrooms", 1, 5, 2, key="fp_bathrooms")
        
        with col2:
            fp_area = st.number_input(
                "Total Area (sq ft)",
                min_value=500,
                max_value=50000,
                value=2000,
                key="fp_area"
            )
            fp_style = st.selectbox(
                "House Style",
                ["Modern", "Contemporary", "Traditional", "Minimalist", "Mediterranean", "Victorian"],
                key="fp_style"
            )
        
        if st.button("🎨 Draw Floor Plan", type="primary", use_container_width=True):
            with st.spinner("Creating COMPLETE INTERIOR DESIGN..."):
                fig = planner.draw_interior_design_complete(fp_bedrooms, fp_bathrooms, fp_area, fp_style)
                st.session_state['floor_plan_fig'] = fig
        
        if 'floor_plan_fig' in st.session_state:
            st.success("✅ COMPLETE INTERIOR DESIGN CREATED!")
            st.markdown("---")
            st.pyplot(st.session_state['floor_plan_fig'], use_container_width=True)
            
            st.markdown("""
            ### 📋 COMPLETE HOME INTERIOR DESIGN INCLUDES:
            
            **🏠 LIVING ROOM/HALL:**
            - Comfortable Sofa & Coffee Table
            - TV Unit with Entertainment Center
            - Dining Table with 4 Chairs
            - Wall Decorations & Plants
            
            **🍳 KITCHEN:**
            - 4-Burner Gas Stove
            - Refrigerator for Food Storage
            - Double Sink with Counter
            - Microwave & Storage Cabinets
            - Island Prep Table with Stools
            - Natural Lighting
            
            **🛏️ MASTER BEDROOM:**
            - King-Size Bed with Pillows
            - Bedside Tables with Lamps
            - Large Wardrobe/Closet
            - Study Desk & Chair
            - Lounge/Reading Area
            - Multiple Windows for Light
            
            **🚿 BATHROOM:**
            - Bathtub with Shower Area
            - Modern Shower Cubicle
            - Western Toilet
            - Dual Sink Vanity Counter
            - Large Mirror
            - Storage Cabinet
            - Exhaust Vent
            """)
            
            # Download floor plan image
            buf = io.BytesIO()
            st.session_state['floor_plan_fig'].savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                "💾 Download COMPLETE INTERIOR DESIGN (HIGH QUALITY PNG)",
                buf,
                file_name=f"complete_interior_design_{fp_style}.png",
                mime="image/png",
                use_container_width=True
            )
    
    # Tab 6: SketchUp & 3D Visualization Guide
    with tab6:
        st.header("🏗️ 3D House Visualization")
        st.markdown("Interactive 3D visualization of your home design")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sketch_style = st.selectbox(
                "Architectural Style",
                ["Modern", "Contemporary", "Traditional", "Rustic", "Mediterranean", "Minimalist"],
                key="sketch_style"
            )
        
        with col2:
            sketch_bedrooms = st.slider("Bedrooms (for visualization)", 1, 8, 3, key="sketch_bedrooms")
        
        with col3:
            sketch_view = st.selectbox(
                "View Angle",
                ["Front", "Side", "Top", "Isometric"],
                key="sketch_view"
            )
        
        if st.button("🏘️ Generate 3D View", type="primary", use_container_width=True):
            with st.spinner("Rendering 3D model..."):
                fig_3d = planner.draw_3d_house_view(sketch_bedrooms, sketch_style)
                st.session_state['3d_fig'] = fig_3d
        
        if '3d_fig' in st.session_state:
            st.success("✅ 3D Visualization Created!")
            st.plotly_chart(st.session_state['3d_fig'], use_container_width=True)
            
            st.info("💡 Rotate, zoom, and interact with the 3D model above!")
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
    <small>🏗️ Built with Streamlit & Ollama | AI-Powered Civil Home Planning Assistant</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
