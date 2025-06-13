import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import io
import base64
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Excess Inventory Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitInventoryAnalysis:
    def __init__(self):
        # Fixed color scheme
        self.status_colors = {
            'Within Norms': '#28a745',      # Green
            'Excess Inventory': '#007bff',   # Blue
            'Short Inventory': '#dc3545'    # Red
        }
        
        # Initialize session state
        if 'inventory_data' not in st.session_state:
            st.session_state.inventory_data = []
            self.load_sample_data()
        
        if 'tolerance' not in st.session_state:
            st.session_state.tolerance = 30
    
    def safe_float_convert(self, value):
        """Safely convert string to float, handling commas and other formatting"""
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        
        str_value = str(value).strip()
        str_value = str_value.replace(',', '').replace(' ', '')
        
        if str_value.endswith('%'):
            str_value = str_value[:-1]
        
        try:
            return float(str_value)
        except (ValueError, TypeError):
            return 0.0
    
    def safe_int_convert(self, value):
        """Safely convert string to int, handling commas and other formatting"""
        if pd.isna(value) or value == '' or value is None:
            return 0
        
        str_value = str(value).strip()
        str_value = str_value.replace(',', '').replace(' ', '')
        
        try:
            return int(float(str_value))
        except (ValueError, TypeError):
            return 0
    
    def load_sample_data(self):
        """Load sample inventory data with QTY and RM columns"""
        inventory_sample = [
            ["AC0303020106", "FLAT ALUMINIUM PROFILE", "5.230", "4.000", "496"],
            ["AC0303020105", "RAIN GUTTER PROFILE", "8.360", "6.000", "1984"],
            ["AA0106010001", "HYDRAULIC POWER STEERING OIL", "12.500", "10.000", "2356"],
            ["AC0203020077", "Bulb beading LV battery flap", "3.500", "3.000", "248"],
            ["AC0303020104", "L- PROFILE JAM PILLAR", "15.940", "20.000", "992"],
            ["AA0112014000", "Conduit Pipe Filter to Compressor", "25", "30", "1248"],
            ["AA0115120001", "HVPDU ms", "18", "12", "1888"],
            ["AA0119020017", "REAR TURN INDICATOR", "35", "40", "1512"],
            ["AA0119020019", "REVERSING LAMP", "28", "20", "1152"],
            ["AA0822010800", "SIDE DISPLAY BOARD", "42", "50", "2496"],
            ["BB0101010001", "ENGINE OIL FILTER", "65", "45", "1300"],
            ["BB0202020002", "BRAKE PAD SET", "22", "25", "880"],
            ["CC0303030003", "CLUTCH DISC", "8", "12", "640"],
            ["DD0404040004", "SPARK PLUG", "45", "35", "450"],
            ["EE0505050005", "AIR FILTER", "30", "28", "600"],
            ["FF0606060006", "FUEL FILTER", "55", "50", "1100"],
            ["GG0707070007", "TRANSMISSION OIL", "40", "35", "800"],
            ["HH0808080008", "COOLANT", "22", "30", "660"],
            ["II0909090009", "BRAKE FLUID", "15", "12", "300"],
            ["JJ1010101010", "WINDSHIELD WASHER", "33", "25", "495"]
        ]
        
        # Load inventory data
        st.session_state.inventory_data = []
        for row in inventory_sample:
            st.session_state.inventory_data.append({
                'Material': row[0],
                'Description': row[1],
                'QTY': self.safe_float_convert(row[2]),
                'RM IN QTY': self.safe_float_convert(row[3]),
                'Stock_Value': self.safe_int_convert(row[4])
            })
    
    def calculate_inventory_status(self, current_qty, rm_qty, tolerance):
        """Calculate variance and determine inventory status"""
        if rm_qty == 0:
            variance_pct = 0 if current_qty == 0 else 100
            variance_qty = current_qty
        else:
            variance_pct = ((current_qty - rm_qty) / rm_qty) * 100
            variance_qty = current_qty - rm_qty
        
        # Determine status based on tolerance
        if variance_pct > tolerance:
            status = "Excess Inventory"
        elif variance_pct < -tolerance:
            status = "Short Inventory"
        else:
            status = "Within Norms"
        
        return variance_pct, variance_qty, status
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded CSV or Excel file"""
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype=str)
            elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, sheet_name=0, dtype=str)
            else:
                raise ValueError(f"Unsupported file format")
            
            return self.standardize_inventory_data(df.to_dict('records'))
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return []
    
    def standardize_inventory_data(self, records):
        """Standardize inventory data and extract QTY and RM columns"""
        if not records:
            return []
        
        standardized_data = []
        
        # Find required columns (case insensitive)
        qty_columns = ['qty', 'quantity', 'current_qty', 'stock_qty']
        rm_columns = ['rm', 'rm_qty', 'required_qty', 'norm_qty', 'target_qty', 'rm_in_qty', 'ri_in_qty']
        material_columns = ['material', 'material_code', 'part_number', 'item_code', 'code', 'part_no']
        desc_columns = ['description', 'item_description', 'part_description', 'desc']
        value_columns = ['stock_value', 'value', 'amount', 'cost']
        
        # Get column names (case insensitive)
        sample_item = records[0]
        available_columns = {k.lower().replace(' ', '_'): k for k in sample_item.keys()}
        
        # Find the best matching columns
        qty_col = None
        rm_col = None
        material_col = None
        desc_col = None
        value_col = None
        
        for col_name in qty_columns:
            if col_name in available_columns:
                qty_col = available_columns[col_name]
                break
        
        for col_name in rm_columns:
            if col_name in available_columns:
                rm_col = available_columns[col_name]
                break
        
        for col_name in material_columns:
            if col_name in available_columns:
                material_col = available_columns[col_name]
                break
        
        for col_name in desc_columns:
            if col_name in available_columns:
                desc_col = available_columns[col_name]
                break
        
        for col_name in value_columns:
            if col_name in available_columns:
                value_col = available_columns[col_name]
                break
        
        if not qty_col:
            raise ValueError("QTY/Quantity column not found")
        if not rm_col:
            raise ValueError("RM/RM IN QTY column not found")
        if not material_col:
            raise ValueError("Material/Part Number column not found")
        
        # Process each record
        for record in records:
            try:
                material = str(record.get(material_col, '')).strip()
                qty = self.safe_float_convert(record.get(qty_col, 0))
                rm = self.safe_float_convert(record.get(rm_col, 0))
                
                if material and material.lower() != 'nan' and qty >= 0 and rm >= 0:
                    item = {
                        'Material': material,
                        'Description': str(record.get(desc_col, '')).strip() if desc_col else '',
                        'QTY': qty,
                        'RM IN QTY': rm,
                        'Stock_Value': self.safe_int_convert(record.get(value_col, 0)) if value_col else 0
                    }
                    standardized_data.append(item)
            except Exception:
                continue
        
        return standardized_data
    
    def get_analysis_data(self):
        """Get processed analysis data"""
        analysis_data = []
        summary_data = {
            'Within Norms': {'count': 0, 'value': 0},
            'Excess Inventory': {'count': 0, 'value': 0},
            'Short Inventory': {'count': 0, 'value': 0}
        }
        
        for item in st.session_state.inventory_data:
            variance_pct, variance_qty, status = self.calculate_inventory_status(
                item['QTY'], item['RM IN QTY'], st.session_state.tolerance
            )
            
            analysis_item = {
                'Material': item['Material'],
                'Description': item['Description'],
                'QTY': item['QTY'],
                'RM IN QTY': item['RM IN QTY'],
                'Variance_%': variance_pct,
                'Variance_Qty': variance_qty,
                'Status': status,
                'Stock_Value': item['Stock_Value']
            }
            analysis_data.append(analysis_item)
            
            # Update summary
            summary_data[status]['count'] += 1
            summary_data[status]['value'] += item['Stock_Value']
        
        return analysis_data, summary_data
    
    def create_status_pie_chart(self, summary_data):
        """Create pie chart showing status distribution"""
        labels = list(summary_data.keys())
        values = [summary_data[status]['count'] for status in labels]
        colors = [self.status_colors[status] for status in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Inventory Status Distribution",
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    def create_stock_value_chart(self, analysis_data):
        """Create bar chart showing top 10 parts by stock value"""
        # Sort by stock value and get top 10
        sorted_data = sorted(analysis_data, key=lambda x: x['Stock_Value'], reverse=True)[:10]
        
        materials = [item['Material'] for item in sorted_data]
        values = [item['Stock_Value'] for item in sorted_data]
        colors = [self.status_colors[item['Status']] for item in sorted_data]
        
        fig = go.Figure(data=[go.Bar(
            x=materials,
            y=values,
            marker_color=colors,
            text=[f'₹{v:,}' for v in values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Top 10 Parts by Stock Value",
            xaxis_title="Material Code",
            yaxis_title="Stock Value (₹)",
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_variance_chart(self, analysis_data):
        """Create bar chart showing top 10 materials by variance"""
        # Sort by absolute variance and get top 10
        sorted_data = sorted(analysis_data, key=lambda x: abs(x['Variance_%']), reverse=True)[:10]
        
        materials = [item['Material'] for item in sorted_data]
        variances = [item['Variance_%'] for item in sorted_data]
        colors = [self.status_colors[item['Status']] for item in sorted_data]
        
        fig = go.Figure(data=[go.Bar(
            x=materials,
            y=variances,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in variances],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Top 10 Materials by Variance",
            xaxis_title="Material Code",
            yaxis_title="Variance %",
            height=400,
            xaxis_tickangle=-45
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        return fig
    
    def create_comparison_chart(self, analysis_data):
        """Create grouped bar chart comparing QTY vs RM"""
        # Get top 10 by stock value
        sorted_data = sorted(analysis_data, key=lambda x: x['Stock_Value'], reverse=True)[:10]
        
        materials = [item['Material'] for item in sorted_data]
        qty_values = [item['QTY'] for item in sorted_data]
        rm_values = [item['RM IN QTY'] for item in sorted_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current QTY',
            x=materials,
            y=qty_values,
            marker_color='#17a2b8',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='RM IN QTY',
            x=materials,
            y=rm_values,
            marker_color='#ffc107',
            opacity=0.8
        ))
        
        fig.update_layout(
            title="QTY vs RM IN QTY Comparison",
            xaxis_title="Material Code",
            yaxis_title="Quantity",
            height=400,
            barmode='group',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_scatter_plot(self, analysis_data):
        """Create scatter plot of QTY vs RM IN QTY"""
        qty_values = [item['QTY'] for item in analysis_data]
        rm_values = [item['RM IN QTY'] for item in analysis_data]
        colors = [self.status_colors[item['Status']] for item in analysis_data]
        materials = [item['Material'] for item in analysis_data]
        
        fig = go.Figure()
        
        # Group by status for legend
        for status in self.status_colors.keys():
            status_data = [item for item in analysis_data if item['Status'] == status]
            if status_data:
                fig.add_trace(go.Scatter(
                    x=[item['RM IN QTY'] for item in status_data],
                    y=[item['QTY'] for item in status_data],
                    mode='markers',
                    name=status,
                    marker=dict(color=self.status_colors[status], size=8),
                    text=[item['Material'] for item in status_data],
                    hovertemplate='<b>%{text}</b><br>RM IN QTY: %{x}<br>Current QTY: %{y}<extra></extra>'
                ))
        
        # Add diagonal line
        max_val = max(max(qty_values) if qty_values else 0, max(rm_values) if rm_values else 0)
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Match',
            line=dict(dash='dash', color='black'),
            opacity=0.5
        ))
        
        fig.update_layout(
            title="QTY vs RM IN QTY Scatter Plot",
            xaxis_title="RM IN QTY",
            yaxis_title="Current QTY",
            height=400
        )
        
        return fig
    
    def create_variance_histogram(self, analysis_data):
        """Create histogram of variance distribution"""
        variances = [item['Variance_%'] for item in analysis_data]
        
        fig = go.Figure(data=[go.Histogram(
            x=variances,
            nbinsx=20,
            marker_color='#6c757d',
            opacity=0.7
        )])
        
        fig.add_vline(x=st.session_state.tolerance, line_dash="dash", line_color="red", 
                     annotation_text=f"+{st.session_state.tolerance}%")
        fig.add_vline(x=-st.session_state.tolerance, line_dash="dash", line_color="red",
                     annotation_text=f"-{st.session_state.tolerance}%")
        fig.add_vline(x=0, line_color="green", annotation_text="Perfect Match")
        
        fig.update_layout(
            title="Variance Distribution",
            xaxis_title="Variance %",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    def create_stock_impact_chart(self, summary_data):
        """Create chart showing stock value impact by status"""
        statuses = list(summary_data.keys())
        values = [summary_data[status]['value'] for status in statuses]
        colors = [self.status_colors[status] for status in statuses]
        
        fig = go.Figure(data=[go.Bar(
            x=statuses,
            y=values,
            marker_color=colors,
            text=[f'₹{v:,}' for v in values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Stock Value Impact by Status",
            xaxis_title="Status",
            yaxis_title="Stock Value (₹)",
            height=400
        )
        
        return fig

def main():
    # Initialize the app
    app = StreamlitInventoryAnalysis()
    
    # App header
    st.title("📊 Excess Inventory Analysis")
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Control Panel")
        
        # Tolerance setting
        tolerance = st.selectbox(
            "Tolerance Zone (+/-)",
            options=[10, 20, 30, 40, 50],
            index=2,  # Default to 30%
            format_func=lambda x: f"{x}%"
        )
        st.session_state.tolerance = tolerance
        
        # File upload
        st.subheader("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose inventory file (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File must contain QTY, RM IN QTY, and Material columns"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                new_data = app.process_uploaded_file(uploaded_file)
                if new_data:
                    st.session_state.inventory_data = new_data
                    st.success(f"Loaded {len(new_data)} items")
                else:
                    st.error("Failed to load data")
        
        # Sample data button
        if st.button("🔄 Load Sample Data"):
            app.load_sample_data()
            st.success("Sample data loaded")
        
        st.markdown("---")
        
        # Status criteria
        st.subheader("📋 Status Criteria")
        st.markdown(f"""
        **Within Norms:** QTY = RM IN QTY ± {tolerance}%  
        **Excess Inventory:** QTY > RM IN QTY + {tolerance}%  
        **Short Inventory:** QTY < RM IN QTY - {tolerance}%
        """)
    
    # Main content
    if not st.session_state.inventory_data:
        st.warning("No data available. Please upload a file or load sample data.")
        return
    
    # Get analysis data
    analysis_data, summary_data = app.get_analysis_data()
    
    # Summary dashboard
    st.subheader("📈 Summary Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Within Norms",
            value=f"{summary_data['Within Norms']['count']} parts",
            delta=f"₹{summary_data['Within Norms']['value']:,}"
        )
    
    with col2:
        st.metric(
            label="Excess Inventory",
            value=f"{summary_data['Excess Inventory']['count']} parts",
            delta=f"₹{summary_data['Excess Inventory']['value']:,}"
        )
    
    with col3:
        st.metric(
            label="Short Inventory",
            value=f"{summary_data['Short Inventory']['count']} parts",
            delta=f"₹{summary_data['Short Inventory']['value']:,}"
        )
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Charts", "📋 Data Table", "📥 Export"])
    
    with tab1:
        st.subheader("Graphical Analysis")
        
        # Graph selection
        col1, col2 = st.columns(2)
        
        with col1:
            show_pie = st.checkbox("Status Distribution (Pie Chart)", value=True)
            show_stock_value = st.checkbox("Top 10 Parts by Stock Value", value=True)
            show_variance = st.checkbox("Top 10 Materials by Variance", value=True)
            show_comparison = st.checkbox("QTY vs RM Comparison", value=True)
        
        with col2:
            show_scatter = st.checkbox("QTY vs RM Scatter Plot", value=False)
            show_histogram = st.checkbox("Variance Distribution", value=False)
            show_impact = st.checkbox("Stock Value Impact", value=False)
        
        # Display selected charts
        chart_col1, chart_col2 = st.columns(2)
        
        chart_count = 0
        if show_pie:
            with chart_col1 if chart_count % 2 == 0 else chart_col2:
                st.plotly_chart(app.create_status_pie_chart(summary_data), use_container_width=True)
            chart_count += 1
        
        if show_stock_value:
            with chart_col1 if chart_count % 2 == 0 else chart_col2:
                st.plotly_chart(app.create_stock_value_chart(analysis_data), use_container_width=True)
            chart_count += 1
        
        if show_variance:
            with chart_col1 if chart_count % 2 == 0 else chart_col2:
                st.plotly_chart(app.create_variance_chart(analysis_data), use_container_width=True)
            chart_count += 1
        
        if show_comparison:
            with chart_col1 if chart_count % 2 == 0 else chart_col2:
                st.plotly_chart(app.create_comparison_chart(analysis_data), use_container_width=True)
            chart_count += 1
        
        if show_scatter:
            with chart_col1 if chart_count % 2 == 0 else chart_col2:
                st.plotly_chart(app.create_scatter_plot(analysis_data), use_container_width=True)
            chart_count += 1
        
        if show_histogram:
            with chart_col1 if chart_count % 2 == 0 else chart_col2:
                st.plotly_chart(app.create_variance_histogram(analysis_data), use_container_width=True)
            chart_count += 1
        
        if show_impact:
            with chart_col1 if chart_count % 2 == 0 else chart_col2:
                st.plotly_chart(app.create_stock_impact_chart(summary_data), use_container_width=True)
            chart_count += 1
    
    with tab2:
        st.subheader("Detailed Analysis")
        
        # Create DataFrame for display
        df_display = pd.DataFrame(analysis_data)
        df_display['Variance_%'] = df_display['Variance_%'].round(2)
        df_display['Variance_Qty'] = df_display['Variance_Qty'].round(2)
        df_display['Stock_Value'] = df_display['Stock_Value'].apply(lambda x: f"₹{x:,}")
        
        # Status filter
        status_filter = st.multiselect(
            "Filter by Status:",
            options=['Within Norms', 'Excess Inventory', 'Short Inventory'],
            default=['Within Norms', 'Excess Inventory', 'Short Inventory']
        )
        
        if status_filter:
            filtered_df = df_display[df_display['Status'].isin(status_filter)]
        else:
            filtered_df = df_display
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"inventory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Export Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Excel Report")
            st.markdown("Download comprehensive Excel report with multiple sheets")
            
            if st.button("Generate Excel Report"):
                # Create Excel file in memory
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Detailed analysis sheet
                    df_export = pd.DataFrame(analysis_data)
                    df_export.to_excel(writer, sheet_name='Detailed Analysis', index=False)
                    
                    # Summary sheet
                    summary_export = []
                    for status, data in summary_data.items():
                        summary_export.append({
                            'Status': status,
                            'Part Count': data['count'],
                            'Stock Value': data['value']
                        })
                    
                    pd.DataFrame(summary_export).to_excel(writer, sheet_name='Summary', index=False)
                
                # Create download button
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"inventory_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            st.markdown("### 📋 Summary Report")
            st.markdown("Download summary statistics")
            
            # Create summary text report
            summary_text = f"""
INVENTORY ANALYSIS SUMMARY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tolerance Setting: ±{st.session_state.tolerance}%

OVERALL STATISTICS:
Total Parts Analyzed: {len(analysis_data)}
Total Stock Value: ₹{sum(item['Stock_Value'] for item in analysis_data):,}

STATUS BREAKDOWN:
Within Norms: {summary_data['Within Norms']['count']} parts (₹{summary_data['Within Norms']['value']:,})
Excess Inventory: {summary_data['Excess Inventory']['count']} parts (₹{summary_data['Excess Inventory']['value']:,})
Short Inventory: {summary_data['Short Inventory']['count']} parts (₹{summary_data['Short Inventory']['value']:,})

TOP 5 HIGHEST VARIANCE PARTS:
"""
            
            # Add top 5 variance parts
            top_variance = sorted(analysis_data, key=lambda x: abs(x['Variance_%']), reverse=True)[:5]
            for i, item in enumerate(top_variance, 1):
                summary_text += f"{i}. {item['Material']}: {item['Variance_%']:.1f}% ({item['Status']})\n"
            
            st.download_button(
                label="📥 Download Summary Report",
                data=summary_text,
                file_name=f"inventory_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
