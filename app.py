import random
import time

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np



def create_dataframe_from_lists(dates, causes, resolutions, assignments, ticket_ids, descriptions=None):
    """
    Create a DataFrame from input lists.

    Parameters:
    - dates: list of dates
    - causes: list of high level causes
    - resolutions: list of resolution types
    - assignments: list of assignment groups
    - ticket_ids: list of ticket IDs
    - descriptions: list of short descriptions (optional)

    Returns:
    - pandas DataFrame
    """
    data = {
        'Number': ticket_ids,
        'Date': dates,
        'High Level Cause': causes,
        'Resolution Type': resolutions,
        'Assignment Group': assignments,
    }

    if descriptions is not None:
        data['Short Description'] = descriptions

    return pd.DataFrame(data)


def create_sample_data():
    """Create sample data for testing"""
    causes = ['Software', 'Unknown', 'Hardware', 'Procedural', 'Environmental',
              'Security Event', 'Data', '3rd Party Service Outage', 'Others']
    resolutions = ['Replaced', 'Not listed', 'Removed Transaction', 'Repaired',
                   'Rebooted/Restarted', 'No fault found', 'Repaired (no parts used)',
                   'Resolved without action', 'Others']
    assignments = ['Fujitsu', 'NCR', 'Group 1', 'Others', 'Group 2', 'Group 3']

    # Generate 1000 sample records
    n_records = random.choice([1000, 1100, 1250, 1600, 2000])
    start_date = datetime.now() - timedelta(days=365)

    dates = [start_date + timedelta(days=i % 365) for i in range(n_records)]
    causes_data = np.random.choice(causes, size=n_records)
    resolutions_data = np.random.choice(resolutions, size=n_records)
    assignments_data = np.random.choice(assignments, size=n_records)
    ticket_ids = [f'TKT{i + 1:06d}' for i in range(n_records)]

    return create_dataframe_from_lists(
        dates=dates,
        causes=causes_data,
        resolutions=resolutions_data,
        assignments=assignments_data,
        ticket_ids=ticket_ids
    )


def create_pie_figure(labels, values, title, colors=None):
    """Create a pie chart figure"""
    if len(labels) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False, font_size=14)]
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            name=title,
            textinfo='percent',
            hoverinfo='label+percent',
            marker_colors=colors
        )
    )
    fig.update_layout(
        title=title,
        showlegend=True,
        height=400
    )
    return fig


@st.cache_data
def filter_dataframe(df, causes, resolutions, date_range):
    """Filter DataFrame based on selected filters"""
    filtered_df = df.copy()

    if causes:
        filtered_df = filtered_df[filtered_df['High Level Cause'].isin(causes)]
    if resolutions:
        filtered_df = filtered_df[filtered_df['Resolution Type'].isin(resolutions)]
    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) &
            (filtered_df['Date'] <= end_date)
            ]

    return filtered_df


def create_dashboard(df, unique_id):
    """
    Create the dashboard with the provided DataFrame

    Parameters:
    - df: pandas DataFrame containing the data
    - unique_id: unique identifier for widget keys
    """
    try:
        # Create filters
        col1, col2, col3, col4 = st.columns([1, 1, 1, 0.5])

        with col1:
            causes = st.multiselect(
                "**High Level Cause**",
                options=sorted(df['High Level Cause'].unique()),
                key=f"cause_filter_{unique_id}"
            )

        with col2:
            resolutions = st.multiselect(
                "**Resolution Type**",
                options=sorted(df['Resolution Type'].unique()),
                key=f"resolution_filter_{unique_id}"
            )

        with col3:
            default_dates = (datetime.now() - timedelta(days=30), datetime.now())
            date_range = st.date_input(
                "**Date Range**",
                value=default_dates,
                key=f"date_filter_{unique_id}"
            )

        with col4:
            st.write("")  # Add spacing
            st.write("")  # Add spacing
            if st.button("**Reset Dates**", key=f"reset_button_{unique_id}"):
                date_range = default_dates

        # Filter DataFrame
        filtered_df = filter_dataframe(df, causes, resolutions, date_range)

        # Display record count
        st.metric("Record Count", len(filtered_df))

        # Create summary data for charts
        cause_summary = filtered_df['High Level Cause'].value_counts()
        resolution_summary = filtered_df['Resolution Type'].value_counts()
        assignment_summary = filtered_df['Assignment Group'].value_counts()

        # Create charts
        charts_col1, charts_col2, charts_col3 = st.columns(3)

        # Define color schemes
        cause_colors = ['#1E90FF', '#87CEEB', '#FF4500', '#FFA07A', '#98FB98',
                        '#90EE90', '#FFA500', '#FFD700', '#9370DB']
        resolution_colors = ['#D3D3D3', '#9370DB', '#1E90FF', '#FFB6C1', '#DDA0DD',
                             '#98FB98', '#6495ED', '#90EE90', '#FFA07A']
        assignment_colors = ['#FFE4B5', '#9370DB', '#483D8B', '#9370DB', '#1E90FF', '#87CEEB']

        with charts_col1:
            fig1 = create_pie_figure(
                cause_summary.index,
                cause_summary.values,
                "High Level Cause",
                colors=cause_colors
            )
            st.plotly_chart(fig1, use_container_width=True)

        with charts_col2:
            fig2 = create_pie_figure(
                resolution_summary.index,
                resolution_summary.values,
                "Resolution Type",
                colors=resolution_colors
            )
            st.plotly_chart(fig2, use_container_width=True)

        with charts_col3:
            fig3 = create_pie_figure(
                assignment_summary.index,
                assignment_summary.values,
                "Assignment Group",
                colors=assignment_colors
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Display scrollable dataframe
        st.markdown("### Summary")
        st.dataframe(
            filtered_df.sort_values('Date', ascending=False),
            use_container_width=True,
            height=300
        )

    except Exception as e:
        st.error(f"An error occurred while creating the dashboard: {str(e)}")


def process_query(user_input):
    """Process user input to determine if charts should be shown"""
    if user_input == "Hi":
        time.sleep(5)
    keywords = ['chart', 'charts', 'dashboard', 'visualization', 'visualize', 'graph', 'graphs', 'show me', 'display']
    user_input = user_input.lower()
    return any(keyword in user_input for keyword in keywords)


def get_unique_id():
    """Generate a unique identifier for widgets"""
    st.session_state.widget_counter += 1
    return f"widget_{st.session_state.widget_counter}"


def main():
    """Main function for standalone testing"""
    st.set_page_config(layout="wide")
    st.title("Support Metrics Dashboard")


    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("show_dashboard", False):
                create_dashboard(st.session_state.df, f"hist_{idx}")

    # Chat input
    if prompt := st.chat_input("How can I help you with the support metrics?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query and respond
        show_dashboard = process_query(prompt)

        # Generate assistant response
        if show_dashboard:
            response = "Here's the support metrics dashboard you requested:"
        else:
            response = ("I can help you analyze the support metrics. "
                        "You can ask me to show you the dashboard or charts, "
                        "or ask specific questions about the metrics.")

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            if show_dashboard:
                unique_id = get_unique_id()
                create_dashboard(st.session_state.df, unique_id)

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "show_dashboard": show_dashboard
        })


# def main():
#     """Main function for standalone testing"""
#     st.set_page_config(layout="wide")
#     st.title("Support Metrics Dashboard")
#
#     # Create sample data for testing
#     df = create_sample_data()
#
#     # Create dashboard
#     create_dashboard(df)

if __name__ == "__main__":
    main()
