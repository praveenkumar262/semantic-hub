import streamlit as st
import pandas as pd
from databricks import sql
import requests
import json
import time
import re

# Page configuration
st.set_page_config(
    page_title="Semantic Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #006E33;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #033c67;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #f18a31;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'connection_verified' not in st.session_state:
    st.session_state.connection_verified = False
if 'metadata_df' not in st.session_state:
    st.session_state.metadata_df = None
if 'metadata_text' not in st.session_state:
    st.session_state.metadata_text = None
if 'tmdl_code' not in st.session_state:
    st.session_state.tmdl_code = None
if 'partitions' not in st.session_state:
    st.session_state.partitions = None
if 'databricks_endpoint' not in st.session_state:
    st.session_state.databricks_endpoint = None
if 'headers' not in st.session_state:
    st.session_state.headers = None

def initialize_llm_config(workspace_url, token, model_name="databricks-meta-llama-3-3-70b-instruct"):
    """Initialize LLM configuration"""
    st.session_state.databricks_endpoint = f"{workspace_url}/serving-endpoints/{model_name}/invocations"
    st.session_state.headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def call_llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.1):
    """Call Databricks LLM endpoint"""
    # Add validation
    if st.session_state.databricks_endpoint is None or st.session_state.headers is None:
        return "[Error: LLM endpoint not initialized. Please check connection settings.]"
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(
            st.session_state.databricks_endpoint,
            headers=st.session_state.headers,
            data=json.dumps(payload),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            elif "predictions" in result:
                return result["predictions"][0]
            elif "content" in result:
                return result["content"]
            else:
                return str(result)
        else:
            return f"[Error: Status {response.status_code}] {response.text}"
    except Exception as e:
        return f"[Exception: {str(e)}]"

def test_connection(server_hostname, http_path, access_token, schema_name):
    """Test Databricks connection"""
    try:
        conn = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token
        )
        
        cursor = conn.cursor()
        cursor.execute(f"""
        SELECT table_catalog, table_schema, table_name, table_type
        FROM system.information_schema.tables
        WHERE table_schema = '{schema_name}'
        """)
        
        tables = cursor.fetchall()
        
        if not tables:
            cursor.close()
            conn.close()
            return False, "No tables found", None
        
        cursor.execute(f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema_name}'
        ORDER BY table_name, ordinal_position
        """)
        
        metadata = cursor.fetchall()
        cursor.close()
        conn.close()
        
        metadata_df = pd.DataFrame(
            [(row.table_name, row.column_name, row.data_type) for row in metadata],
            columns=['table_name', 'column_name', 'data_type']
        )
        
        return True, f"Connected! Found {len(tables)} tables", metadata_df
        
    except Exception as e:
        return False, f"Connection failed: {str(e)}", None

def generate_metadata(metadata_df):
    """Generate analytical metadata"""
    all_descriptions = []
    tables = metadata_df['table_name'].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, table_name in enumerate(tables):
        status_text.text(f"Analyzing {table_name} ({idx + 1}/{len(tables)})...")
        
        table_data = metadata_df[metadata_df['table_name'] == table_name]
        columns_str = "\n".join(
            [f"  - {row['column_name']}: {row['data_type']}" for _, row in table_data.iterrows()]
        )
        
        prompt = f"""You are a data modeling expert specializing in semantic models and TMDL (Tabular Model Definition Language) design.

Analyze the following table and describe its analytical purpose, relationships, and metric potential. 
Avoid special characters or markdown. Use plain text.

Table Name: {table_name}
Column Details:
{columns_str}

Provide a structured analytical description that can directly support automated TMDL generation, covering the following aspects:

1. Analytical Purpose: Explain how this table contributes to analytics (e.g., reporting, dimensional relationships, aggregation, trend analysis, etc.)
2. Column Roles: For each column, specify its analytical role — whether it is a Dimension, Metric (Measure), Identifier, or Attribute.
   - Clearly mention which columns are used for aggregation or calculations.
3. Keys: Identify the Primary Key and possible Foreign Keys with expected join relationships.
4. Analytical Relationships: Describe how this table can connect to other tables.
   - Mention potential join paths (e.g., can join to 'Sales' via 'Customer_ID').
5. Common Query Patterns: Describe typical ways analysts would query or aggregate data from this table.
6. Metrics Potential: List columns that can be converted into Measures in TMDL (e.g., Total_Sales, Revenue, Quantity_Sold).
7. Column-Level Descriptions: For each column, provide a concise one-line analytical description including its data type and role.

Output the description in clean, readable plain text — suitable for another model to directly generate a TMDL model including relationships and measures.
Do not include any special characters like equals, hash, asterisk, dash, or backticks."""
        
        description = call_llm(prompt, max_tokens=700, temperature=0.7)
        table_section = f"Table Name: {table_name}\n{description}\n"
        all_descriptions.append(table_section)
        
        progress_bar.progress((idx + 1) / len(tables))
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.success("Metadata generation complete!")
    return "\n".join(all_descriptions)

def map_datatype(dtype):
    """Map source datatype to TMDL"""
    dtype = dtype.lower()
    if any(x in dtype for x in ['int', 'long', 'bigint']):
        return "int64", "0", "sum"
    elif any(x in dtype for x in ['string', 'varchar', 'char', 'text']):
        return "string", "", "none"
    elif any(x in dtype for x in ['double', 'float', 'decimal']):
        return "double", "#,##0.00", "sum"
    elif 'date' in dtype:
        return "dateTime", "Short Date", "none"
    elif 'timestamp' in dtype:
        return "dateTime", "General Date", "none"
    else:
        return "string", "", "none"

def generate_table_tmdl(table_name, metadata_df):
    """Generate TMDL for one table"""
    table_data = metadata_df[metadata_df['table_name'] == table_name]
    
    lines = [f"\ttable {table_name}", ""]
    
    for _, row in table_data.iterrows():
        col_name = row['column_name']
        dtype, fmt, sumby = map_datatype(row['data_type'])
        
        lines.append(f"\t\tcolumn {col_name}")
        lines.append(f"\t\t\tdataType: {dtype}")
        if fmt:
            lines.append(f"\t\t\tformatString: {fmt}")
        lines.append(f"\t\t\tsummarizeBy: {sumby}")
        lines.append(f"\t\t\tsourceColumn: {col_name}")
        lines.append("")
        lines.append(f"\t\t\tannotation SummarizationSetBy = Automatic")
        lines.append("")
    
    lines.append(f"\t\tpartition {table_name} = m")
    lines.append("\t\t\tmode: import")
    lines.append(f"\t\t\tsource = <PARTITION_{table_name}>")
    lines.append("")
    lines.append("\t\tannotation PBI_ResultType = Table")
    
    return '\n'.join(lines)

def generate_complete_tmdl(metadata_text, metadata_df, partitions):
    """Generate complete TMDL"""
    
    # Tables
    progress = st.progress(0)
    status = st.empty()
    
    status.info(" Step 1/5: Generating tables...")
    tables = []
    for table in metadata_df['table_name'].unique():
        tables.append(generate_table_tmdl(table, metadata_df))
    
    tables_code = "\n\n".join(tables)
    progress.progress(0.2)
    
    # Replace partitions
    status.info(" Step 2/5: Applying partitions...")
    for table, partition in partitions.items():
        tables_code = tables_code.replace(
            f"\t\t\tsource = <PARTITION_{table}>",
            f"\t\t\tsource =\n{partition}"
        )
    progress.progress(0.4)
    
    # Relationships
    status.info(" Step 3/5: Generating relationships...")
    rel_prompt = f"""Generate TMDL relationships from metadata:
{metadata_text[:2000]}

Format:
\trelationship Name
\t\tfromColumn: FactTable.FK
\t\ttoColumn: DimTable.PK

Only output TMDL code."""
    
    relationships = call_llm(rel_prompt, 1500)
    progress.progress(0.6)
    
    # Measures
    status.info(" Step 4/5: Generating measures...")
    meas_prompt = f"""Generate TMDL measures from metadata:
{metadata_text[:2000]}

Format:
\tref table TableName

\t\tmeasure Name = DAX
\t\t\tformatString: format
\t\t\tdisplayFolder: folder

Only output TMDL code."""
    
    measures = call_llm(meas_prompt, 2000)
    progress.progress(0.8)
    
    # Assemble
    status.info(" Step 5/5: Assembling final TMDL...")
    final = f"""createOrReplace

{tables_code}

{relationships}

{measures}"""
    
    progress.progress(1.0)
    status.success("✅ TMDL generation complete!")
    time.sleep(1)
    progress.empty()
    status.empty()
    
    return final

# Main App
st.markdown('<div class="main-header">🎯 Semantic Hub</div>', unsafe_allow_html=True)
st.markdown("### AI powered semantic model")

# Sidebar
with st.sidebar:
    st.header("Progress")
    steps = [
        (" Database Connection", 1),
        (" Metadata Generation", 2),
        (" Partition Configuration", 3),
        (" TMDL Generation", 4),
        (" Export & Instructions", 5)
    ]
    
    for step_name, step_num in steps:
        if step_num < st.session_state.step:
            st.success(step_name)
        elif step_num == st.session_state.step:
            st.info(step_name)
        else:
            st.text(step_name)
    
    st.markdown("---")
    if st.button("🔄 Reset Application", key="sidebar_reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Step 1: Connection
if st.session_state.step == 1:
    st.markdown('<div class="step-header">Step 1: Database Connection</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        server = st.text_input("Server Hostname", "dbc-14f40aec-e3df.cloud.databricks.com")
        http_path = st.text_input("HTTP Path", "/sql/1.0/warehouses/e736bec7c6523985")
    
    with col2:
        token = st.text_input("Access Token", type="password")
        schema = st.text_input("Schema Name", "new_one")
    
    if st.button("🔗 Test Connection", type="primary", key="test_conn_btn"):
        if all([server, http_path, token, schema]):
            with st.spinner("Testing connection..."):
                workspace_url = f"https://{server}"
                initialize_llm_config(workspace_url, token)
                
                success, msg, metadata_df = test_connection(server, http_path, token, schema)
                
                if success:
                    st.session_state.connection_verified = True
                    st.session_state.metadata_df = metadata_df
                    st.session_state.server = server
                    st.session_state.http_path = http_path
                    st.session_state.token = token
                    st.session_state.schema = schema
                    
                    st.success(f"✅ {msg}")
                    
                    # Display tables
                    st.subheader("Available Tables")
                    tables = metadata_df['table_name'].unique()
                    
                    # Create grid layout for tables
                    num_cols = min(len(tables), 4)
                    cols = st.columns(num_cols)
                    for i, table in enumerate(tables):
                        with cols[i % num_cols]:
                            table_count = len(metadata_df[metadata_df['table_name'] == table])
                            st.metric(table, f"{table_count} cols")
                else:
                    st.error(f"❌ {msg}")
        else:
            st.error("❌ Please fill in all connection details")
    
    # Show continue button only after successful connection
    if st.session_state.connection_verified:
        st.markdown("---")
        if st.button("➡️ Continue to Metadata Generation", type="primary", key="continue_step1"):
            st.session_state.step = 2
            st.rerun()

# Step 2: Metadata
elif st.session_state.step == 2:
    st.markdown('<div class="step-header">Step 2: Metadata Generation</div>', unsafe_allow_html=True)
    
    if st.session_state.metadata_text is None:
        st.info("Ready to generate analytical metadata using AI...")
        
        if st.button("Generate Metadata", type="primary", key="gen_metadata_btn"):
            with st.spinner("Analyzing tables and generating metadata..."):
                metadata_text = generate_metadata(st.session_state.metadata_df)
                st.session_state.metadata_text = metadata_text
            st.rerun()
    else:
        st.success("✅ Metadata generation complete!")
        
        with st.expander("📄 View Generated Metadata", expanded=False):
            st.text_area("Metadata", st.session_state.metadata_text, height=400, key="metadata_view")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Connection", key="back_step2"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("➡️ Continue to Partition Configuration", type="primary", key="continue_step2"):
                st.session_state.step = 3
                st.rerun()

# Step 3: Partitions
elif st.session_state.step == 3:
    st.markdown('<div class="step-header">Step 3: Partition Configuration</div>', unsafe_allow_html=True)
    
    tables = st.session_state.metadata_df['table_name'].unique()
    total_tables = len(tables)
    
    # Instructions Section
    st.markdown("""
    <div class="info-box">
    <strong> Instructions</strong><br><br>
    Below are the tables detected from your schema. For each table, you need to provide the <strong>Partition Source</strong> (M Query) that defines how Power BI will connect to and load data from that table.
    <br><br>
    <strong>customer_table</strong><br>
    <strong>Partition source:</strong> let     Source = DatabricksMultiCloud.Catalogs("dbc-14f40aec-e3df.cloud.databricks.com", "/sql/1.0/warehouses/e736bec7c6523985", [Catalog = "", Database = ""]),     workspace_Database = Source{[Name="workspace",Kind="Database"]}[Data],     new_one_Schema = workspace_Database{[Name="new_one",Kind="Schema"]}[Data],     customer_table_Table = new_one_Schema{[Name="customer_table",Kind="Table"]}[Data] in     customer_table_Table
    <br><br>
    <strong>fact_order_data_20_k</strong><br>
    <strong>Partition source:</strong> let     Source = DatabricksMultiCloud.Catalogs("dbc-14f40aec-e3df.cloud.databricks.com", "/sql/1.0/warehouses/e736bec7c6523985", [Catalog = "", Database = ""]),     workspace_Database = Source{[Name="workspace",Kind="Database"]}[Data],     new_one_Schema = workspace_Database{[Name="new_one",Kind="Schema"]}[Data],     fact_order_data_20_k_Table = new_one_Schema{[Name="fact_order_data_20_k",Kind="Table"]}[Data] in     fact_order_data_20_k_Table
    <br><br>
    <strong>feedback_table</strong><br>
    <strong>Partition source:</strong> let     Source = DatabricksMultiCloud.Catalogs("dbc-14f40aec-e3df.cloud.databricks.com", "/sql/1.0/warehouses/e736bec7c6523985", [Catalog = "", Database = ""]),     workspace_Database = Source{[Name="workspace",Kind="Database"]}[Data],     new_one_Schema = workspace_Database{[Name="new_one",Kind="Schema"]}[Data],     feedback_table_Table = new_one_Schema{[Name="feedback_table",Kind="Table"]}[Data] in     feedback_table_Table
    <br><br>
    <strong>food_table</strong><br>
    <strong>Partition source:</strong> let     Source = DatabricksMultiCloud.Catalogs("dbc-14f40aec-e3df.cloud.databricks.com", "/sql/1.0/warehouses/e736bec7c6523985", [Catalog = "", Database = ""]),     workspace_Database = Source{[Name="workspace",Kind="Database"]}[Data],     new_one_Schema = workspace_Database{[Name="new_one",Kind="Schema"]}[Data],     food_table_Table = new_one_Schema{[Name="food_table",Kind="Table"]}[Data] in     food_table_Table
    <br><br>
    <strong>restaurent_table</strong><br>
    <strong>Partition source:</strong> let     Source = DatabricksMultiCloud.Catalogs("dbc-14f40aec-e3df.cloud.databricks.com", "/sql/1.0/warehouses/e736bec7c6523985", [Catalog = "", Database = ""]),     workspace_Database = Source{[Name="workspace",Kind="Database"]}[Data],     new_one_Schema = workspace_Database{[Name="new_one",Kind="Schema"]}[Data],     restaurent_table_Table = new_one_Schema{[Name="restaurent_table",Kind="Table"]}[Data] in     restaurent_table_Table                             
    </div>
    """, unsafe_allow_html=True)
    
    # Display detected tables
    st.markdown("###  Detected Tables")
    st.markdown(f"**Total Tables:** {total_tables}")
    
    # Show table list
    table_list = ", ".join([f"`{table}`" for table in tables])
    st.markdown(f"**Tables:** {table_list}")
    
    st.markdown("---")
    st.markdown("###  Configure Partition Sources")
    st.markdown("Enter or modify the partition source (M Query) for each table below:")
    
    # Initialize partitions if not exists
    if st.session_state.partitions is None:
        st.session_state.partitions = {}
    
    # Template for Databricks connection
    template = f"""let
    Source = DatabricksMultiCloud.Catalogs("{st.session_state.server}", "{st.session_state.http_path}", [Catalog = "", Database = ""]),
    workspace_Database = Source{{[Name="workspace",Kind="Database"]}}[Data],
    {st.session_state.schema}_Schema = workspace_Database{{[Name="{st.session_state.schema}",Kind="Schema"]}}[Data],
    TABLE_Table = {st.session_state.schema}_Schema{{[Name="TABLE",Kind="Table"]}}[Data]
in
    TABLE_Table"""
    
    # Create partition inputs with counter
    for idx, table in enumerate(tables, 1):
        st.markdown(f"#### **{table}**")
        st.markdown("**Partition Source:**")
        
        # Generate table-specific source
        default_source = template.replace("TABLE", table)
        
        # Store partition source in session state
        partition_key = f"partition_{table}"
        if partition_key not in st.session_state:
            st.session_state[partition_key] = default_source
        
        source = st.text_area(
            f"M Query for {table}",
            value=st.session_state[partition_key],
            height=150,
            key=f"part_input_{table}",
            label_visibility="collapsed",
            help=f"Enter the M Query partition source for {table}. Default Databricks connection is pre-filled."
        )
        
        # Format with proper indentation for TMDL
        formatted = "\n".join(["				" + line.strip() for line in source.split("\n") if line.strip()])
        st.session_state.partitions[table] = formatted
        
        st.markdown("---")
    
    st.markdown("###")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Metadata", key="back_step3"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("➡️ Generate TMDL Code", type="primary", key="continue_step3"):
            st.session_state.step = 4
            st.rerun()

# Step 4: Generate
elif st.session_state.step == 4:
    st.markdown('<div class="step-header">Step 4: TMDL Generation</div>', unsafe_allow_html=True)
    
    if st.session_state.tmdl_code is None:
        st.info(" Ready to generate complete TMDL code...")
        
        if st.button(" Generate TMDL", type="primary", key="gen_tmdl_btn"):
            tmdl = generate_complete_tmdl(
                st.session_state.metadata_text,
                st.session_state.metadata_df,
                st.session_state.partitions
            )
            st.session_state.tmdl_code = tmdl
            st.rerun()
    else:
        st.success("✅ TMDL code generation complete!")
        
        st.subheader(" Generated TMDL Code")
        
        # Show preview
        preview_length = min(3000, len(st.session_state.tmdl_code))
        st.code(st.session_state.tmdl_code[:preview_length] + "\n\n... (code continues) ...", language="")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Length", f"{len(st.session_state.tmdl_code)} chars")
        with col2:
            st.metric("Tables", len(st.session_state.metadata_df['table_name'].unique()))
        with col3:
            st.metric("Total Columns", len(st.session_state.metadata_df))
        
        # Download button
        st.download_button(
            label="⬇️ Download Complete TMDL File",
            data=st.session_state.tmdl_code,
            file_name="semantic_model.tmdl",
            mime="text/plain",
            type="primary"
        )
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Configuration", key="back_step4"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("➡️ View Power BI Instructions", type="primary", key="continue_step4"):
                st.session_state.step = 5
                st.rerun()

# Step 5: Instructions
elif st.session_state.step == 5:
    st.markdown('<div class="step-header">Step 5: Power BI Integration</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h3> Your TMDL Code is Ready!</h3>
    <p>Follow these steps to create your AI-powered semantic model in Power BI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download button at top
    st.download_button(
        label="⬇️ Download semantic_model.tmdl",
        data=st.session_state.tmdl_code,
        file_name="semantic_model.tmdl",
        mime="text/plain",
        type="primary",
        key="download_final"
    )
    
    st.markdown("""
    ###  Implementation Steps
    
    #### Method 1: Using Tabular Editor (Recommended)
    
    **Step 1: Install Tabular Editor 3**
    - Download from: https://tabulareditor.com/
    - Install and launch the application
    
    **Step 2: Connect to Power BI**
    - Open Power BI Desktop
    - Create a new report or open existing
    - Go to **External Tools** ribbon → Click **Tabular Editor 3**
    
    **Step 3: Import TMDL**
    - In Tabular Editor: **File** → **Open** → **From File**
    - Select your downloaded `semantic_model.tmdl`
    - Review the imported structure
    
    **Step 4: Save to Power BI**
    - In Tabular Editor: **File** → **Save**
    - Changes are automatically applied to Power BI
    
    **Step 5: Configure Databricks Connection**
    - In Power BI Desktop: **Transform Data** → **Data Source Settings**
    - Update Databricks credentials
    - Test connection
    
    **Step 6: Refresh Data**
    - Click **Refresh** in Power BI
    - Verify all tables load correctly
    
    **Step 7: Publish**
    - **File** → **Publish** → **Publish to Power BI**
    - Select your workspace
    - Configure scheduled refresh in Power BI Service
    
    ---
    
    #### Method 2: Using XMLA Endpoint
    
    **Prerequisites:**
    - Power BI Premium or Premium Per User
    - XMLA Read-Write enabled in workspace settings
    
    **Steps:**
    1. Enable XMLA endpoint in Power BI Service workspace settings
    2. Use Tabular Editor to connect via XMLA
    3. Import the TMDL file
    4. Deploy directly to Power BI Service
    
    ---
    
    ### ✅ Verification Checklist
    
    - ✅ All tables are imported
    - ✅ Relationships are correctly defined
    - ✅ Measures are working
    - ✅ Data refreshes successfully
    - ✅ Databricks credentials configured
    
    ---
    
    ###  Additional Resources
    
    - [TMDL Documentation](https://learn.microsoft.com/en-us/analysis-services/tmdl/tmdl-overview)
    - [Tabular Editor Docs](https://docs.tabulareditor.com/)
    - [Power BI Semantic Models](https://learn.microsoft.com/en-us/power-bi/connect-data/service-datasets-understand)
    - [Databricks Power BI Integration](https://docs.databricks.com/integrations/bi/power-bi.html)
    
    ---
    
    ###  Important Notes
    
    - **Backup First**: Always backup existing Power BI models before importing
    - **Test Environment**: Test with sample data first
    - **Credentials**: Ensure Databricks access token is valid and has proper permissions
    - **Version Control**: Save TMDL files in Git for tracking changes
    - **Review Measures**: Validate all DAX measures before production use
    
    ---
    
    ###  Pro Tips
    
    - Use **Tabular Editor 3** for best TMDL support
    - Enable **Enhanced Metadata Format** in Power BI Desktop for better compatibility
    - Keep your TMDL files in version control (Git) for collaboration
    - Document custom measures and relationships for your team
    - Set up incremental refresh for large datasets
    
    """)
    
    st.markdown("""
    <div class="warning-box">
    <strong>🔒 Security Reminder:</strong> Never commit access tokens to version control. Use environment variables or secure vaults for credentials.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Back to TMDL", key="back_step5"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("🔄 Start New Project", key="new_project"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col3:
        st.download_button(
            label="⬇️ Download Again",
            data=st.session_state.tmdl_code,
            file_name="semantic_model.tmdl",
            mime="text/plain",
            key="download_again"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Semantic Hub v1.0</strong> | Powered by Databricks & AI</p>
    <p>Need help? Contact your administrator</p>
</div>
""", unsafe_allow_html=True)