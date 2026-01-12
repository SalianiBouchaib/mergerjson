import streamlit as st
import pandas as pd
import json
from typing import Any, Dict, List, Union
from collections import defaultdict
import io

def flatten_json(data: Any, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Recursively flatten nested JSON structures
    Handles nested dicts, lists, and mixed structures
    """
    items = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(flatten_json(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                if len(value) == 0:
                    items.append((new_key, ''))
                elif all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                    items.append((new_key, ', '.join(str(v) for v in value if v is not None)))
                else:
                    for idx, item in enumerate(value):
                        indexed_key = f"{new_key}_{idx}"
                        if isinstance(item, (dict, list)):
                            items. extend(flatten_json(item, indexed_key, sep=sep).items())
                        else:  
                            items.append((indexed_key, str(item) if item is not None else ''))
            else:
                items. append((new_key, str(value) if value is not None else ''))
    
    elif isinstance(data, list):
        if len(data) == 0:
            items.append((parent_key, ''))
        elif all(isinstance(item, (str, int, float, bool, type(None))) for item in data):
            items.append((parent_key, ', '.join(str(v) for v in data if v is not None)))
        else:
            for idx, item in enumerate(data):
                indexed_key = f"{parent_key}_{idx}" if parent_key else str(idx)
                if isinstance(item, (dict, list)):
                    items.extend(flatten_json(item, indexed_key, sep=sep).items())
                else:
                    items.append((indexed_key, str(item) if item is not None else ''))
    
    else:
        items.append((parent_key, str(data) if data is not None else ''))
    
    return dict(items)

def extract_all_keys(data: Any, parent_key: str = '', sep: str = '_') -> set:
    """
    Extract all possible keys from JSON structure to create complete column set
    """
    keys = set()
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            keys.add(new_key)
            
            if isinstance(value, (dict, list)):
                keys. update(extract_all_keys(value, new_key, sep=sep))
    
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, (dict, list)):
                indexed_key = f"{parent_key}_{idx}" if parent_key else str(idx)
                keys.update(extract_all_keys(item, indexed_key, sep=sep))
    
    return keys

def json_to_dataframe(json_data: Union[Dict, List], flatten: bool = True, separator: str = '_') -> pd.DataFrame:
    """
    Convert JSON to DataFrame with complete flattening
    """
    if isinstance(json_data, dict):
        if flatten:
            flattened = flatten_json(json_data, sep=separator)
            df = pd.DataFrame([flattened])
        else:
            df = pd.DataFrame([json_data])
    
    elif isinstance(json_data, list):
        if flatten:
            all_keys = set()
            for item in json_data:
                all_keys. update(extract_all_keys(item, sep=separator))
            
            flattened_items = []
            for item in json_data:
                flattened = flatten_json(item, sep=separator)
                for key in all_keys:
                    if key not in flattened:  
                        flattened[key] = ''
                flattened_items.append(flattened)
            
            df = pd.DataFrame(flattened_items)
        else:
            df = pd.DataFrame(json_data)
    
    else:
        df = pd.DataFrame([{'value': json_data}])
    
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.fillna('')
    
    return df

def analyze_json_structure(data: Any, level: int = 0) -> Dict[str, Any]:  
    """
    Analyze JSON structure to provide insights
    """
    analysis = {
        'type': type(data).__name__,
        'level': level,
        'size': 0,
        'keys': [],
        'nested_structures': []
    }
    
    if isinstance(data, dict):
        analysis['size'] = len(data)
        analysis['keys'] = list(data.keys())
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                nested_analysis = analyze_json_structure(value, level + 1)
                analysis['nested_structures']. append({
                    'key': key,
                    'analysis': nested_analysis
                })
    
    elif isinstance(data, list):
        analysis['size'] = len(data)
        if len(data) > 0:
            analysis['item_types'] = list(set(type(item).__name__ for item in data))
            if isinstance(data[0], dict):
                analysis['keys'] = list(data[0]. keys())
    
    return analysis

def safe_json_loads(json_string: str) -> tuple:
    """
    Safely load JSON with error handling - supports regular JSON and JSONL (newline-delimited)
    """
    # Try regular JSON first
    try: 
        data = json.loads(json_string)
        return data, None, 'json'
    except json.JSONDecodeError as e:
        # If regular JSON fails, try JSONL (JSON Lines / newline-delimited JSON)
        try:
            lines = json_string.strip().split('\n')
            json_objects = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        obj = json.loads(line)
                        json_objects.append(obj)
                    except json. JSONDecodeError as line_error:
                        return None, f"JSONL Parse Error on line {line_num}: {str(line_error)}", None
            
            if json_objects:
                return json_objects, None, 'jsonl'
            else:
                return None, "No valid JSON objects found", None
        
        except Exception as jsonl_error:
            return None, f"JSON Parse Error: {str(e)}\n\nAlso tried JSONL format:  {str(jsonl_error)}", None
    
    except Exception as e:  
        return None, f"Error:  {str(e)}", None

def main():
    st.set_page_config(
        page_title="JSON to CSV - Ultimate Extractor",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 JSON to CSV Converter - Ultimate Extractor")
    st.markdown("Convert JSON to CSV with **complete flattening** - extracts EVERYTHING from nested structures")
    
    with st.sidebar:
        st.header("📖 About")
        st.info(
            "**Ultimate JSON to CSV Extractor**\n\n"
            "**Features:**\n"
            "- Deep nested object flattening\n"
            "- Array handling (indexed)\n"
            "- Mixed data type support\n"
            "- All keys extraction\n"
            "- Structure analysis\n"
            "- Multiple input methods\n"
            "- JSONL support (newline-delimited)\n\n"
            "**Handles:**\n"
            "- Single JSON objects\n"
            "- JSON arrays\n"
            "- JSONL/NDJSON (one JSON per line)\n"
            "- Deeply nested structures\n"
            "- Arrays within objects\n"
            "- Complex hierarchies\n"
            "- Missing fields"
        )
        
        st.header("⚙️ Settings")
        flatten_nested = st.checkbox("Flatten nested structures", value=True, help="Convert nested JSON to flat columns")
        separator = st.text_input("Key separator", value="_", help="Character to separate nested keys")
        show_empty_cols = st.checkbox("Show empty columns", value=False)
        
        st.header("📊 Options")
        show_analysis = st.checkbox("Show JSON structure analysis", value=True)
        show_raw_preview = st.checkbox("Show raw JSON preview", value=False)
    
    st.subheader("📥 Input Method")
    input_method = st.radio(
        "Choose input method:",
        ["Upload JSON file", "Paste JSON text"],
        horizontal=True
    )
    
    json_data = None
    data_source = None
    json_format = None
    
    if input_method == "Upload JSON file":    
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json', 'txt', 'jsonl', 'ndjson'],
            help="Upload a JSON, JSONL, or text file containing JSON data"
        )
        
        if uploaded_file is not None:  
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 File", uploaded_file.name)
            with col2:
                file_size_kb = uploaded_file.size / 1024
                st. metric("💾 Size", f"{file_size_kb:.2f} KB")
            with col3:
                st.metric("📝 Type", uploaded_file.type if uploaded_file.type else "JSON")
            
            try: 
                json_string = uploaded_file.read().decode('utf-8')
                json_data, error, json_format = safe_json_loads(json_string)
                
                if error:
                    st.error(f"❌ {error}")
                    st.info("💡 **Tip**:  If you have multiple JSON objects, make sure they are:\n"
                           "- Wrapped in an array:  `[{...}, {...}]`\n"
                           "- OR on separate lines (JSONL format)")
                else:
                    data_source = uploaded_file.name
                    format_name = "JSONL (JSON Lines)" if json_format == 'jsonl' else "JSON"
                    st.success(f"✅ {format_name} file loaded successfully!")
            
            except Exception as e:  
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif input_method == "Paste JSON text":  
        st.markdown("**Paste your JSON data below:**")
        
        json_text = st.text_area(
            "JSON Data",
            height=300,
            placeholder='{\n  "key": "value",\n  "nested": {\n    "data": "here"\n  }\n}\n\nOR (JSONL format):\n{"id": 1, "name": "Alice"}\n{"id": 2, "name":  "Bob"}',
            label_visibility="collapsed"
        )
        
        if json_text. strip():
            if st.button("🔄 Parse JSON", type="primary"):
                json_data, error, json_format = safe_json_loads(json_text)
                
                if error:
                    st.error(f"❌ {error}")
                    st.info("💡 **Tip**: If you have multiple JSON objects, make sure they are:\n"
                           "- Wrapped in an array:  `[{...}, {...}]`\n"
                           "- OR on separate lines (JSONL format)")
                else: 
                    data_source = "Pasted JSON"
                    format_name = "JSONL (JSON Lines)" if json_format == 'jsonl' else "JSON"
                    st.success(f"✅ {format_name} parsed successfully!")
    
    if json_data is not None: 
        st.divider()
        
        if show_analysis:
            st.subheader("🔍 JSON Structure Analysis")
            
            analysis = analyze_json_structure(json_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📦 Type", analysis['type'])
            with col2:
                st.metric("📊 Size", analysis['size'])
            with col3:
                if isinstance(json_data, list) and len(json_data) > 0:
                    st. metric("📋 Records", len(json_data))
                elif isinstance(json_data, dict):
                    st.metric("🔑 Keys", len(json_data))
                else:
                    st.metric("🔢 Items", 1)
            with col4:
                nested_count = len(analysis. get('nested_structures', []))
                format_display = "JSONL" if json_format == 'jsonl' else "JSON"
                st.metric("📄 Format", format_display)
            
            with st.expander("📋 Detailed Structure"):
                if isinstance(json_data, dict):
                    st.write("**Top-level keys:**")
                    st.write(analysis['keys'])
                    
                    if analysis['nested_structures']:
                        st.write("**Nested structures:**")
                        for nested in analysis['nested_structures']:  
                            st.write(f"- `{nested['key']}` ({nested['analysis']['type']})")
                
                elif isinstance(json_data, list):
                    st.write(f"**Array with {len(json_data)} items**")
                    if 'item_types' in analysis:     
                        st.write(f"**Item types:** {', '.join(analysis['item_types'])}")
                    if 'keys' in analysis:  
                        st.write("**Common keys:**")
                        st. write(analysis['keys'])
        
        if show_raw_preview:  
            with st.expander("📄 Raw JSON Preview"):
                if isinstance(json_data, list) and len(json_data) > 10:
                    st.json(json_data[:10])
                    st.caption(f"Showing first 10 of {len(json_data)} items")
                else:
                    st.json(json_data)
        
        st.subheader("🔄 Converting to CSV")
        
        with st.spinner("Extracting all data and flattening structures..."):
            try:
                df = json_to_dataframe(json_data, flatten=flatten_nested, separator=separator)
                
                st.success(f"✅ Converted to DataFrame:  {len(df)} rows × {len(df.columns)} columns")
                
                st.session_state['df'] = df
                st.session_state['data_source'] = data_source
            
            except Exception as e:  
                st.error(f"❌ Conversion error: {str(e)}")
                st.exception(e)
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        if not show_empty_cols:
            df_display = df.loc[:, (df != '').any(axis=0)]
        else:
            df_display = df
        
        st.divider()
        
        st. subheader("📈 Extraction Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📋 Rows", len(df))
        with col2:
            st.metric("📊 Total Columns", len(df.columns))
        with col3:
            st.metric("✅ Non-Empty Columns", len(df_display.columns))
        with col4:
            non_empty_cells = (df != '').sum().sum()
            st.metric("📦 Non-Empty Cells", non_empty_cells)
        with col5:
            fill_rate = (non_empty_cells / (len(df) * len(df. columns)) * 100) if len(df) > 0 and len(df.columns) > 0 else 0
            st.metric("💯 Fill Rate", f"{fill_rate:.1f}%")
        
        st.subheader("📊 Column Analysis")
        
        col_stats = []
        for col in df.columns:
            non_empty = (df[col] != '').sum()
            col_stats.append({
                'Column': col,
                'Non-Empty': non_empty,
                'Empty': len(df) - non_empty,
                'Fill %': f"{(non_empty / len(df) * 100):.1f}%" if len(df) > 0 else "0%",
                'Unique Values': df[col].nunique(),
                'Data Type': str(df[col].dtype)
            })
        
        col_stats_df = pd.DataFrame(col_stats)
        
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox(
                "Sort columns by:",
                ['Column', 'Non-Empty', 'Fill %', 'Unique Values'],
                index=1
            )
        with col2:
            filter_empty = st.checkbox("Hide empty columns", value=False)
        
        if filter_empty:
            col_stats_df = col_stats_df[col_stats_df['Non-Empty'] > 0]
        
        if sort_by == 'Fill %':
            col_stats_df = col_stats_df. sort_values('Non-Empty', ascending=False)
        else:
            col_stats_df = col_stats_df.sort_values(sort_by, ascending=False if sort_by != 'Column' else True)
        
        st.dataframe(col_stats_df, use_container_width=True, height=300)
        
        st.subheader("👁️ Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_rows = st.slider("Rows to display", 5, 100, 10)
        with col2:
            search_term = st.text_input("🔍 Search in data", "")
        with col3:
            column_filter = st.multiselect(
                "Select specific columns",
                options=list(df_display.columns),
                default=[]
            )
        
        filtered_df = df_display. copy()
        
        if search_term:
            mask = filtered_df. apply(lambda row: row. astype(str).str.contains(search_term, case=False).any(), axis=1)
            filtered_df = filtered_df[mask]
        
        if column_filter:
            filtered_df = filtered_df[column_filter]
        
        st.dataframe(filtered_df. head(num_rows), use_container_width=True, height=400)
        
        with st.expander("📊 Value Distribution (for selected columns)"):
            if len(df_display. columns) > 0:
                selected_col = st.selectbox("Select column to analyze:", df_display.columns)
                
                value_counts = df_display[selected_col].value_counts().head(20)
                
                col1, col2 = st. columns(2)
                with col1:
                    st.bar_chart(value_counts)
                with col2:
                    st.dataframe(
                        value_counts.reset_index().rename(columns={'index': 'Value', selected_col: 'Count'}),
                        use_container_width=True
                    )
        
        st.subheader("💾 Download CSV Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                csv_all = df.to_csv(index=False, encoding='utf-8-sig', quoting=1)
                st.download_button(
                    label="📥 Full CSV (All Columns)",
                    data=csv_all,
                    file_name=f"full_{st.session_state. get('data_source', 'data')}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                st.caption(f"✅ {len(df. columns)} columns")
            except Exception as e:  
                st.error(f"Error:  {e}")
        
        with col2:
            try:
                df_non_empty = df. loc[:, (df != '').any(axis=0)]
                csv_filtered = df_non_empty.to_csv(index=False, encoding='utf-8-sig', quoting=1)
                st. download_button(
                    label="📥 Filtered CSV (Non-Empty)",
                    data=csv_filtered,
                    file_name=f"filtered_{st.session_state.get('data_source', 'data')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"✅ {len(df_non_empty.columns)} columns")
            except Exception as e:     
                st.error(f"Error: {e}")
        
        with col3:
            try:
                if column_filter:
                    csv_custom = filtered_df.to_csv(index=False, encoding='utf-8-sig', quoting=1)
                    st.download_button(
                        label="📥 Custom CSV (Selected)",
                        data=csv_custom,
                        file_name=f"custom_{st.session_state.get('data_source', 'data')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.caption(f"✅ {len(filtered_df.columns)} columns")
                else:  
                    st.info("Select columns above to enable custom download")
            except Exception as e:   
                st.error(f"Error: {e}")
        
        with st.expander("⚙️ Advanced Export Options"):
            st.markdown("**Additional export formats:**")
            
            col1, col2 = st. columns(2)
            
            with col1:
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df. to_excel(writer, index=False, sheet_name='Data')
                    
                    st.download_button(
                        label="📊 Download as Excel",
                        data=buffer.getvalue(),
                        file_name=f"{st.session_state.get('data_source', 'data')}. xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except:    
                    st.warning("Excel export requires:  pip install openpyxl")
            
            with col2:
                json_output = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📄 Download as JSON",
                    data=json_output,
                    file_name=f"{st.session_state.get('data_source', 'data')}_converted.json",
                    mime="application/json"
                )
        
        st.info(f"📊 Displaying {len(filtered_df)} of {len(df)} total rows")
    
    with st.expander("❓ Help & Examples"):
        st.markdown("""
        ### How to use:
        
        1. **Upload or paste your JSON data**
        2. **Configure settings** (flattening, separator)
        3. **Review the structure analysis** to understand your data
        4. **Preview the extracted data**
        5. **Download CSV** with all extracted fields
        
        ### Supported JSON formats:
        
        **1. Regular JSON object:**
        ```json
        {
          "name": "John",
          "age": 30,
          "city": "New York"
        }
        ```
        
        **2. JSON array:**
        ```json
        [
          {"id": 1, "name": "John"},
          {"id": 2, "name": "Jane"}
        ]
        ```
        
        **3. JSONL (JSON Lines) - one object per line:**
        ```
        {"id": 1, "name": "John", "age": 30}
        {"id": 2, "name": "Jane", "age": 25}
        {"id": 3, "name":  "Bob", "age": 35}
        ```
        
        **4. Nested structures:**
        ```json
        {
          "user": {
            "profile": {
              "name": "John",
              "contacts": [
                {"type": "email", "value": "john@example.com"},
                {"type": "phone", "value": "555-1234"}
              ]
            }
          }
        }
        ```
        
        ### Features:
        - **Deep nesting**:  Unlimited nesting levels
        - **Arrays**: Indexed extraction (item_0, item_1, etc.)
        - **JSONL support**: Newline-delimited JSON
        - **Mixed types**: Handles strings, numbers, booleans, nulls
        - **Missing fields**: Automatically fills with empty values
        - **All keys**: Extracts every possible field across all records
        """)

if __name__ == "__main__":
    main()