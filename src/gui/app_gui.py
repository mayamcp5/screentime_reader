import streamlit as st
import pandas as pd
from pathlib import Path
import io
import base64
from PIL import Image
import zipfile
import tempfile

from src.ios.overall import process_ios_overall_screenshot
from src.ios.activity import process_ios_category_screenshot
from src.android.overall import process_android_overall_screenshot
from src.android.activity_history import process_android_activity_history

def main():
    # Page config
    st.set_page_config(
        page_title="Screen Time Extractor",
        page_icon="⏱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # UNC Color Theme
    UNC_BLUE = "#13294B"
    UNC_LIGHT_BLUE = "#7BAFD4"
    UNC_WHITE = "#FFFFFF"
    UNC_GRAY = "#F0F0F0"

    # Custom CSS
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        * {{
            font-family: 'Inter', sans-serif;
        }}
        
        /* Hide default Streamlit header padding */
        .main > div {{
            padding-top: 0rem;
        }}
        
        .main-header {{
            background: linear-gradient(135deg, {UNC_BLUE} 0%, {UNC_LIGHT_BLUE} 100%);
            color: white;
            padding: 2rem 4rem;
            margin: -1rem -4rem 2rem -4rem;
            width: calc(100% + 8rem);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
            width: 100vw;
        }}
        
        .header-content {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }}
        
        .header-subtitle {{
            font-size: 1.1rem;
            font-weight: 400;
            opacity: 0.95;
            margin-top: 0.5rem;
            letter-spacing: 0.3px;
        }}
        
        .stButton>button {{
            background-color: {UNC_BLUE};
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        .stButton>button:hover {{
            background-color: {UNC_LIGHT_BLUE};
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
            background-color: {UNC_GRAY};
            padding: 0.5rem;
            border-radius: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: white;
            border-radius: 6px;
            font-weight: 600;
            color: {UNC_BLUE};
        }}
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: {UNC_BLUE};
            color: white;
        }}
        
        .warning-box {{
            background-color: #FFF3CD;
            border-left: 4px solid #FFC107;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            color: #856404;
        }}
        
        .info-box {{
            background-color: #D1ECF1;
            border-left: 4px solid #17A2B8;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            color: #0C5460;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Helper functions
    def format_for_excel(data, result_type):
        """Format data for Excel paste (tab-separated)"""
        lines = []
        
        if result_type in ['ios_overall', 'android_overall']:
            lines.append(f"Date\t{data.get('date', 'N/A')}")
            lines.append(f"Total Time\t{data.get('total_time', '0h 0m')}")
            lines.append("")
        
        if data.get('categories'):
            lines.append("Top Categories")
            lines.append("Category\tTime")
            for c in data['categories']:
                lines.append(f"{c['name']}\t{c['time']}")
            lines.append("")
        
        if data.get('top_apps'):
            lines.append("Top Apps")
            lines.append("App\tTime")
            for a in data['top_apps']:
                lines.append(f"{a['name']}\t{a['time']}")
            lines.append("")
        
        if data.get('hourly_usage'):
            lines.append("Hourly Usage")
            lines.append("Hour\tOverall\tSocial\tEntertainment")
            for hour in ['12am','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am',
                        '12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm']:
                if hour in data['hourly_usage']:
                    h = data['hourly_usage'][hour]
                    lines.append(f"{hour}\t{h.get('overall',0)}\t{h.get('social',0)}\t{h.get('entertainment',0)}")
            lines.append("")
        
        if data.get('apps'):
            category = data.get('category', 'Activity')
            lines.append(f"{category} Apps")
            lines.append("App\tTime")
            for a in data['apps']:
                lines.append(f"{a['name']}\t{a['time']}")
        
        return "\n".join(lines)

    def get_result_label(result_type, data):
        """Get display label for result type"""
        if result_type == "ios_overall":
            return "Overall"
        elif result_type == "android_overall":
            return "Overall"
        elif result_type == "ios_category":
            return data.get('category', 'Category')
        elif result_type == "android_activity":
            return "Activity"
        return "Result"

    def display_result_section(data, result_type, image_path=None, filename=None):
        """Display a single result with image preview and copy buttons"""
        
        # Create columns for image and data
        if image_path:
            if isinstance(image_path, list):
                # Multiple images (Android activity)
                img_col, data_col = st.columns([1, 2])
                with img_col:
                    st.subheader("Screenshots")
                    for idx, img in enumerate(image_path):
                        with st.expander(f"View Image {idx+1}: {Path(img).name}", expanded=False):
                            st.image(img, use_container_width=True)
            else:
                # Single image
                img_col, data_col = st.columns([1, 2])
                with img_col:
                    st.subheader("Screenshot")
                    with st.expander(f"View Image: {filename or 'Screenshot'}", expanded=False):
                        st.image(image_path, use_container_width=True)
        else:
            data_col = st.container()
        
        with data_col:
            # iOS Overall
            if result_type == "ios_overall":
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Date", data.get('date', 'N/A'))
                with col2:
                    st.metric("Total Time", data.get('total_time', '0h 0m'))
                with col3:
                    st.metric("Y-Max (pixels)", data.get('ymax_pixels', 'N/A'))
                
                st.divider()
                
                # Categories
                if data.get('categories'):
                    st.subheader("Top Categories")
                    df = pd.DataFrame(data['categories'])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Copy button for categories
                    cat_text = "Top Categories\n" + "\n".join([f"{c['name']}\t{c['time']}" for c in data['categories']])
                    st.code(cat_text, language=None)
                else:
                    st.markdown('<div class="warning-box">⚠️ No category data available</div>', unsafe_allow_html=True)
                
                # Top Apps
                if data.get('top_apps'):
                    st.subheader("Top Apps")
                    df = pd.DataFrame(data['top_apps'])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Copy button for apps
                    apps_text = "Top Apps\n" + "\n".join([f"{a['name']}\t{a['time']}" for a in data['top_apps']])
                    st.code(apps_text, language=None)
                else:
                    st.markdown('<div class="warning-box">⚠️ No app data extracted</div>', unsafe_allow_html=True)
                
                # Hourly Usage
                if data.get('hourly_usage'):
                    st.subheader("Hourly Usage (pixels)")
                    hourly_data = []
                    for hour in ['12am','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am',
                                '12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm']:
                        if hour in data['hourly_usage']:
                            h_data = data['hourly_usage'][hour]
                            hourly_data.append({
                                'Hour': hour,
                                'Overall': h_data.get('overall', 0),
                                'Social': h_data.get('social', 0),
                                'Entertainment': h_data.get('entertainment', 0)
                            })
                    
                    df = pd.DataFrame(hourly_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Copy button for hourly
                    hourly_text = "Hour\tOverall\tSocial\tEntertainment\n" + "\n".join(
                        [f"{row['Hour']}\t{row['Overall']}\t{row['Social']}\t{row['Entertainment']}" 
                        for _, row in df.iterrows()]
                    )
                    st.code(hourly_text, language=None)
                else:
                    st.markdown('<div class="warning-box">⚠️ No hourly data available</div>', unsafe_allow_html=True)
            
            # iOS Category
            elif result_type == "ios_category":
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Category", data.get('category', 'N/A'))
                with col2:
                    st.metric("Total Time", data.get('total_time', '0h 0m'))
                
                st.divider()
                
                if data.get('apps'):
                    st.subheader(f"{data.get('category', 'Category')} Apps")
                    df = pd.DataFrame(data['apps'])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    apps_text = f"{data.get('category', 'Category')} Apps\n" + "\n".join(
                        [f"{a['name']}\t{a['time']}" for a in data['apps']]
                    )
                    st.code(apps_text, language=None)
                else:
                    st.markdown('<div class="warning-box">⚠️ No apps found in this category</div>', unsafe_allow_html=True)
            
            # Android Overall
            elif result_type == "android_overall":
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Date", data.get('date', 'N/A'))
                with col2:
                    st.metric("Total Time", data.get('total_time', '0h 0m'))
                
                st.divider()
                
                if data.get('top_apps'):
                    st.subheader("Top Apps")
                    df = pd.DataFrame(data['top_apps'])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    apps_text = "Top Apps\n" + "\n".join([f"{a['name']}\t{a['time']}" for a in data['top_apps']])
                    st.code(apps_text, language=None)
                else:
                    st.markdown('<div class="warning-box">⚠️ No app data available</div>', unsafe_allow_html=True)
            
            # Android Activity
            elif result_type == "android_activity":
                if data.get('apps'):
                    st.subheader("Activity Apps")
                    df = pd.DataFrame(data['apps'])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    apps_text = "Activity Apps\n" + "\n".join([f"{a['name']}\t{a['time']}" for a in data['apps']])
                    st.code(apps_text, language=None)
                else:
                    st.markdown('<div class="warning-box">⚠️ No activity data available</div>', unsafe_allow_html=True)
            
            # Copy All button
            st.divider()
            excel_format = format_for_excel(data, result_type)
            st.text_area("Copy All (Excel Format)", excel_format, height=150, 
                        help="Copy this text and paste directly into Excel - it will automatically separate into columns")

    # Header
    st.markdown('''
    <div class="main-header">
        <div class="header-content">
            <div class="header-title">Screen Time Data Extractor</div>
            <div class="header-subtitle">UNC Research Data Processing Tool</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'mode' not in st.session_state:
        st.session_state.mode = 'single'

    # Mode Selection
    st.header("Select Processing Mode")
    mode_cols = st.columns(2)
    with mode_cols[0]:
        if st.button("Single File Mode", use_container_width=True, 
                    type="primary" if st.session_state.mode == 'single' else "secondary"):
            st.session_state.mode = 'single'
            st.session_state.results = []

    with mode_cols[1]:
        if st.button("Batch Processing Mode", use_container_width=True,
                    type="primary" if st.session_state.mode == 'batch' else "secondary"):
            st.session_state.mode = 'batch'
            st.session_state.results = []

    st.divider()

    # SINGLE FILE MODE
    if st.session_state.mode == 'single':
        st.header("Step 1: Select Platform")
        platform = st.radio("", ["iOS", "Android"], horizontal=True, label_visibility="collapsed")
        
        st.header("Step 2: Upload Screenshots")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Screenshots")
            overall_files = st.file_uploader(
                "Upload overall screenshot(s)",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="overall"
            )
        
        with col2:
            category_label = "Activity Screenshots" if platform == "Android" else "Category Screenshots"
            st.subheader(category_label)
            category_files = st.file_uploader(
                f"Upload {category_label.lower()}",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="category"
            )
        
        st.header("Step 3: Process & View Results")
        
        if st.button("Process Screenshots", type="primary", use_container_width=True):
            if not overall_files and not category_files:
                st.error("Please upload at least one screenshot!")
            else:
                st.session_state.results = []
                
                with st.spinner("Processing screenshots..."):
                    try:
                        if platform == "iOS":
                            for uploaded_file in overall_files:
                                temp_path = f"/tmp/{uploaded_file.name}"
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                result = process_ios_overall_screenshot(temp_path)
                                st.session_state.results.append({
                                    "type": "ios_overall",
                                    "name": uploaded_file.name,
                                    "data": result,
                                    "image": temp_path
                                })
                            
                            for uploaded_file in category_files:
                                temp_path = f"/tmp/{uploaded_file.name}"
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                result = process_ios_category_screenshot(temp_path)
                                st.session_state.results.append({
                                    "type": "ios_category",
                                    "name": uploaded_file.name,
                                    "data": result,
                                    "image": temp_path
                                })
                        
                        elif platform == "Android":
                            for uploaded_file in overall_files:
                                temp_path = f"/tmp/{uploaded_file.name}"
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                result = process_android_overall_screenshot(temp_path)
                                st.session_state.results.append({
                                    "type": "android_overall",
                                    "name": uploaded_file.name,
                                    "data": result,
                                    "image": temp_path
                                })
                            
                            if category_files:
                                temp_paths = []
                                for uploaded_file in category_files:
                                    temp_path = f"/tmp/{uploaded_file.name}"
                                    with open(temp_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    temp_paths.append(temp_path)
                                
                                result = process_android_activity_history(temp_paths)
                                st.session_state.results.append({
                                    "type": "android_activity",
                                    "name": "Activity History",
                                    "data": result,
                                    "image": temp_paths  # List of images
                                })
                        
                        st.success(f"Successfully processed {len(st.session_state.results)} file(s)!")
                    
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # BATCH PROCESSING MODE
    else:
        st.header("Batch Processing Mode")
        st.markdown('<div class="info-box">📦 Upload a ZIP file containing folders. Each folder should contain screenshots for one participant (overall file first, then activity/category files).</div>', unsafe_allow_html=True)
        
        zip_file = st.file_uploader("Upload ZIP file with organized folders", type=['zip'])
        platform_batch = st.radio("Select Platform", ["iOS", "Android"], horizontal=True)
        
        if st.button("Process Batch", type="primary", use_container_width=True):
            if not zip_file:
                st.error("Please upload a ZIP file!")
            else:
                st.session_state.results = []
                
                with st.spinner("Processing batch..."):
                    try:
                        # Extract ZIP
                        with tempfile.TemporaryDirectory() as temp_dir:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                zip_ref.extractall(temp_dir)
                            
                            # Process each folder
                            temp_path = Path(temp_dir)
                            folders = [f for f in temp_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
                            
                            for folder in folders:
                                images = sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                                
                                if not images:
                                    continue
                                
                                folder_results = {
                                    "folder_name": folder.name,
                                    "results": [],
                                    "errors": []
                                }
                                
                                try:
                                    if platform_batch == "iOS":
                                        # First file is overall
                                        if len(images) > 0:
                                            result = process_ios_overall_screenshot(str(images[0]))
                                            folder_results["results"].append({
                                                "type": "ios_overall",
                                                "name": images[0].name,
                                                "data": result
                                            })
                                        
                                        # Rest are categories
                                        for img in images[1:]:
                                            result = process_ios_category_screenshot(str(img))
                                            folder_results["results"].append({
                                                "type": "ios_category",
                                                "name": img.name,
                                                "data": result
                                            })
                                    
                                    else:  # Android
                                        if len(images) > 0:
                                            result = process_android_overall_screenshot(str(images[0]))
                                            folder_results["results"].append({
                                                "type": "android_overall",
                                                "name": images[0].name,
                                                "data": result
                                            })
                                        
                                        if len(images) > 1:
                                            result = process_android_activity_history([str(img) for img in images[1:]])
                                            folder_results["results"].append({
                                                "type": "android_activity",
                                                "name": "Activity History",
                                                "data": result
                                            })
                                
                                except Exception as e:
                                    folder_results["errors"].append(str(e))
                                
                                st.session_state.results.append(folder_results)
                        
                        st.success(f"Processed {len(st.session_state.results)} folders!")
                    
                    except Exception as e:
                        st.error(f"Error processing batch: {str(e)}")

    # Display Results
    if st.session_state.results:
        st.divider()
        st.header("Results")
        
        if st.session_state.mode == 'single':
            # Single mode - show tabs
            tabs = st.tabs([get_result_label(r['type'], r['data']) for r in st.session_state.results])
            
            for tab, result in zip(tabs, st.session_state.results):
                with tab:
                    display_result_section(result['data'], result['type'], result.get('image'), result['name'])
            
            # Clear/Reset and Export buttons
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Clear All & Reset", use_container_width=True, type="secondary"):
                    st.session_state.results = []
                    st.rerun()
            
            with col2:
                if st.button("📥 Download All as CSV", use_container_width=True):
                    csv_buffer = io.StringIO()
                    
                    for result in st.session_state.results:
                        csv_buffer.write(f"\n=== {result['name']} ===\n\n")
                        csv_buffer.write(format_for_excel(result['data'], result['type']).replace('\t', ','))
                        csv_buffer.write("\n\n")
                    
                    st.download_button(
                        label="Download CSV File",
                        data=csv_buffer.getvalue(),
                        file_name="screen_time_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        else:
            # Batch mode - show collapsible folders
            for folder_result in st.session_state.results:
                folder_name = folder_result['folder_name']
                has_errors = len(folder_result['errors']) > 0
                
                status_icon = "⚠️" if has_errors else "✅"
                
                with st.expander(f"{status_icon} {folder_name} ({len(folder_result['results'])} files)", expanded=False):
                    if has_errors:
                        st.error(f"Errors: {', '.join(folder_result['errors'])}")
                    
                    for result in folder_result['results']:
                        st.subheader(f"{get_result_label(result['type'], result['data'])} - {result['name']}")
                        display_result_section(result['data'], result['type'], None, result['name'])
                        st.divider()
            
            # Clear and Export buttons
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Clear All & Reset", use_container_width=True, type="secondary", key="batch_clear"):
                    st.session_state.results = []
                    st.rerun()
            
            with col2:
                if st.button("📥 Download Batch Results as CSV", use_container_width=True):
                    csv_buffer = io.StringIO()
                    
                    for folder_result in st.session_state.results:
                        csv_buffer.write(f"\n\n========== {folder_result['folder_name']} ==========\n\n")
                        
                        for result in folder_result['results']:
                            csv_buffer.write(f"\n--- {result['name']} ---\n")
                            csv_buffer.write(format_for_excel(result['data'], result['type']).replace('\t', ','))
                            csv_buffer.write("\n")
                    
                    st.download_button(
                        label="Download Batch CSV File",
                        data=csv_buffer.getvalue(),
                        file_name="batch_screen_time_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # Sidebar
    with st.sidebar:
        st.header("How to Use")
        st.markdown("""
        ### Single File Mode
        1. Select your platform (iOS or Android)
        2. Upload screenshots
        3. Click Process
        4. View results with image preview
        5. Copy sections or download CSV
        
        ### Batch Mode
        1. Organize screenshots in folders
        2. Create ZIP file
        3. Upload and process
        4. Review collapsed results
        5. Export all data
        
        ---
        
        **Tips:**
        - Results show warnings if data is missing
        - Use code boxes to copy individual sections
        - Excel format uses tabs for easy paste
        - Check image preview to verify extraction
        """)
        
        if st.button("Clear All", use_container_width=True):
            st.session_state.results = []
            st.rerun()