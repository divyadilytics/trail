import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px  # Added for interactive visualizations

# Snowflake/Cortex Configuration
HOST = "JJHWDJH-NHB38878.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
# STAGE = "CORTEX_SEARCH"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.propertymanagement"

# Single semantic model
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/grants.yaml'

# Streamlit Page Config
st.set_page_config(
    page_title="Welcome to Cortex AI Assistant ",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []  # Initialize chat history
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
# Initialize chart selection persistence
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"
# Initialize query and results persistence
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None

# Hide Streamlit branding and prevent chat history shading
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
/* Prevent shading of previous chat messages */
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Function to start a new conversation
def start_new_conversation():
    st.session_state.chat_history = []
    st.session_state.current_query = None
    st.session_state.current_results = None
    st.session_state.current_sql = None
    st.session_state.current_summary = None
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.rerun()

# Authentication logic
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI")
    st.markdown("Please login to interact with your data")

    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")

    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user=st.session_state.username,
                password=st.session_state.password,
                account="JJHWDJH-NHB38878",
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH",
                role="ACCOUNTADMIN",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn

            snowpark_session = Session.builder.configs({
                "connection": conn
            }).create()
            st.session_state.snowpark_session = snowpark_session

            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")

            st.session_state.authenticated = True
            st.success("Authentication successful! Redirecting...")
            st.rerun()

        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    session = st.session_state.snowpark_session

    # Utility Functions
    def run_snowflake_query(query):
        try:
            if not query:
                st.warning("⚠️ No SQL query generated.")
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                return None
            columns = df.schema.names
            result_df = pd.DataFrame(data, columns=columns)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            return None

    def is_structured_query(query: str):
        structured_patterns = [
            r'\b(county|number|where|group by|order by|completed units|sum|count|avg|max|min|least|highest|which)\b',
            r'\b(total|how many|leads |profit|projects|jurisdiction|month|year|energy savings|kwh)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str):
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|type of questions|queries)\b.*\b(ask|can i ask|pose)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples|sample questions)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def complete(prompt, model="mistral-large"):
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"❌ COMPLETE Function Error: {str(e)}")
            return None

    def summarize(text):
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"❌ SUMMARIZE Function Error: {str(e)}")
            return None

    def parse_sse_response(response_text: str) -> List[Dict]:
        """Parse SSE response into a list of JSON objects."""
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":  # Skip the [DONE] marker
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}  # Reset for next event
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse SSE data: {str(e)} - Data: {data_str}")
        return events

    def snowflake_api_call(query: str, is_structured: bool = False):
        payload = {
            "model": "mistral-large",
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 1}}

        try:
            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT // 1000
            )
            if st.session_state.debug_mode:  # Show debug info only if toggle is enabled
                st.write(f"API Response Status: {resp.status_code}")
                st.write(f"API Raw Response: {resp.text}")
            if resp.status_code < 400:
                if not resp.text.strip():
                    st.error("❌ API returned an empty response.")
                    return None
                return parse_sse_response(resp.text)
            else:
                raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ API Request Failed: {str(e)}")
            return None

    def summarize_unstructured_answer(answer):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"• {sent.strip()}" for sent in sentences[:6])

    def process_sse_response(response, is_structured):
        sql = ""
        search_results = []
        if not response:
            return sql, search_results
        try:
            for event in response:
                if event.get("event") == "message.delta" and "data" in event:
                    delta = event["data"].get("delta", {})
                    content = delta.get("content", [])
                    for item in content:
                        if item.get("type") == "tool_results":
                            tool_results = item.get("tool_results", {})
                            if "content" in tool_results:
                                for result in tool_results["content"]:
                                    if result.get("type") == "json":
                                        result_data = result.get("json", {})
                                        if is_structured and "sql" in result_data:
                                            sql = result_data.get("sql", "")
                                        elif not is_structured and "searchResults" in result_data:
                                            search_results = [sr["text"] for sr in result_data["searchResults"]]
        except Exception as e:
            st.error(f"❌ Error Processing Response: {str(e)}")
        return sql.strip(), search_results

    # Visualization Function
    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
        """Allows user to select chart options and displays a chart with unique widget keys."""
        if df.empty or len(df.columns) < 2:
            return  # Do not show anything if visualization is not possible

        # Determine default chart type based on query
        query_lower = query.lower()
        if re.search(r'\b(county|jurisdiction)\b', query_lower):
            default_chart = "Pie Chart"
        elif re.search(r'\b(month|year|date)\b', query_lower):
            default_chart = "Line Chart"
        else:
            default_chart = "Bar Chart"

        all_cols = list(df.columns)
        col1, col2, col3 = st.columns(3)

        default_x = st.session_state.get(f"{prefix}_x", all_cols[0])
        try:
            x_index = all_cols.index(default_x)
        except ValueError:
            x_index = 0
        x_col = col1.selectbox("X axis", all_cols, index=x_index, key=f"{prefix}_x")

        remaining_cols = [c for c in all_cols if c != x_col]
        default_y = st.session_state.get(f"{prefix}_y", remaining_cols[0])
        try:
            y_index = remaining_cols.index(default_y)
        except ValueError:
            y_index = 0
        y_col = col2.selectbox("Y axis", remaining_cols, index=y_index, key=f"{prefix}_y")

        chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
        default_type = st.session_state.get(f"{prefix}_type", default_chart)
        try:
            type_index = chart_options.index(default_type)
        except ValueError:
            type_index = chart_options.index(default_chart)
        chart_type = col3.selectbox("Chart Type", chart_options, index=type_index, key=f"{prefix}_type")

        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_line")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_bar")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_pie")
        elif chart_type == "Scatter Chart":
            fig = px.scatter(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_scatter")
        elif chart_type == "Histogram Chart":
            fig = px.histogram(df, x=x_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_hist")

    # UI Logic
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            background-color: #29B5E8 !important;
            color: white !important;
            font-weight: bold !important;
            width: 100% !important;
            border-radius: 0px !important;
            margin: 0 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        logo_container = st.container()
        button_container = st.container()
        about_container = st.container()
        help_container = st.container()

        with logo_container:
            logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
            st.image(logo_url, width=250)

        with button_container:
            st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
            if st.button("New Conversation", key="new_conversation"):
                start_new_conversation()

        with about_container:
            st.markdown("### About")
            st.write(
                "This application uses **Snowflake Cortex Analyst** to interpret "
                "your natural language questions and generate data insights. "
                "Simply ask a question below to see relevant answers and visualizations."
            )

        with help_container:
            st.markdown("### Help & Documentation")
            st.write(
                "- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)  \n"
                "- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/)  \n"
                "- [Contact Support](https://www.snowflake.com/en/support/)"
            )

    st.title("Cortex AI Assistant by DiLytics")

    # Display the fixed semantic model
    semantic_model_filename = SEMANTIC_MODEL.split("/")[-1]
    st.markdown(f"Semantic Model: `{semantic_model_filename}`")

    st.sidebar.subheader("Sample Questions")
    sample_questions = [
        "What is Eco Sustain Innovations?",
        "What is Green Residences program?",
        "Describe the energy savings technologies used in Green Residences.",
        "Show total energy savings by county.",
        "Which county has the highest kWh savings?",
        "How many active projects are there",
        "What is the average kWh savings",
        "Which counties has the min and max of kWh savings",
        "Which counties has the least and highest of kWh savings"
    ]

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "results" in message and message["results"] is not None:
                with st.expander("View SQL Query", expanded=False):
                    st.code(message["sql"], language="sql")
                st.markdown(f"**Query Results ({len(message['results'])} rows):**")
                st.dataframe(message["results"])
                # Only show visualization if it can be rendered
                if not message["results"].empty and len(message["results"].columns) >= 2:
                    st.markdown("**📈 Visualization:**")
                    display_chart_tab(message["results"], prefix=f"chart_{hash(message['content'])}", query=message.get("query", ""))

    query = st.chat_input("Ask your question...")

    for sample in sample_questions:
        if st.sidebar.button(sample, key=sample):
            query = sample

    if query:
        # Reset chart selections for new query
        st.session_state.chart_x_axis = None
        st.session_state.chart_y_axis = None
        st.session_state.chart_type = "Bar Chart"  # Will be overridden in display_chart_tab based on query

        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                is_structured = is_structured_query(query)
                is_complete = is_complete_query(query)
                is_summarize = is_summarize_query(query)
                is_suggestion = is_question_suggestion_query(query)

                assistant_response = {"role": "assistant", "content": "", "query": query}
                if is_suggestion:
                    response_content = "**Here are some questions you can ask me:**\n"
                    for i, q in enumerate(sample_questions, 1):
                        response_content += f"{i}. {q}\n"
                    response_content += "\nFeel free to ask any of these or come up with your own related to energy savings, Green Residences, or other programs!"
                    st.markdown(response_content)
                    assistant_response["content"] = response_content

                elif is_complete:
                    response = complete(query)
                    if response:
                        response_content = f"**✍️ Generated Response:**\n{response}"
                        st.markdown(response_content)
                        assistant_response["content"] = response_content
                    else:
                        response_content = "⚠️ Failed to generate a response."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                elif is_summarize:
                    summary = summarize(query)
                    if summary:
                        response_content = f"**Summary:**\n{summary}"
                        st.markdown(response_content)
                        assistant_response["content"] = response_content
                    else:
                        response_content = "⚠️ Failed to generate a summary."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                elif is_structured:
                    response = snowflake_api_call(query, is_structured=True)
                    sql, _ = process_sse_response(response, is_structured=True)
                    if sql:
                        results = run_snowflake_query(sql)
                        if results is not None and not results.empty:
                            # Convert results to string and use complete function for natural language summary
                            results_text = results.to_string(index=False)
                            prompt = f"Provide a concise natural language answer to the query '{query}' using the following data, avoiding phrases like 'Based on the query results':\n\n{results_text}"
                            summary = complete(prompt)
                            if not summary:
                                summary = "⚠️ Unable to generate a natural language summary."
                            response_content = f"**✍️ Generated Response:**\n{summary}"
                            st.markdown(response_content)
                            with st.expander("View SQL Query", expanded=False):
                                st.code(sql, language="sql")
                            st.markdown(f"**Query Results ({len(results)} rows):**")
                            st.dataframe(results)
                            # Only show visualization if it can be rendered
                            if len(results.columns) >= 2:
                                st.markdown("**📈 Visualization:**")
                                display_chart_tab(results, prefix=f"chart_{hash(query)}", query=query)
                            assistant_response.update({
                                "content": response_content,
                                "sql": sql,
                                "results": results,
                                "summary": summary
                            })
                        else:
                            response_content = "⚠️ No data found."
                            st.warning(response_content)
                            assistant_response["content"] = response_content
                    else:
                        response_content = "⚠️ No SQL generated."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                else:
                    response = snowflake_api_call(query, is_structured=False)
                    _, search_results = process_sse_response(response, is_structured=False)
                    if search_results:
                        raw_result = search_results[0]
                        summary = summarize(raw_result)
                        if summary:
                            response_content = f"**Here is the Answer:**\n{summary}"
                            last_sentence = summary.split(".")[-2] if "." in summary else summary
                            st.markdown(response_content)
                            st.success(f" Key Insight: {last_sentence.strip()}")
                            assistant_response["content"] = response_content
                        else:
                            response_content = f"**🔍 Key Information (Unsummarized):**\n{summarize_unstructured_answer(raw_result)}"
                            st.markdown(response_content)
                            assistant_response["content"] = response_content
                    else:
                        response_content = "⚠️ No relevant search results found."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                # Add assistant response to chat history
                st.session_state.chat_history.append(assistant_response)
                # Update current query and results
                st.session_state.current_query = query
                st.session_state.current_results = assistant_response.get("results")
                st.session_state.current_sql = assistant_response.get("sql")
                st.session_state.current_summary = assistant_response.get("summary")
