# Import libraries

from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
jobs_info = pd.read_csv('Glassdoor_DS_Jobs_NW.csv')

# Create the Dash app
app = Dash()

# Set up the app layout
geo_dropdown = dcc.Dropdown(options=jobs_info['Job Location'].unique(),
                            value='NY')

app.layout = html.Div(children=[
    html.H1(children='Job Salary by States Dashboard'),
    geo_dropdown,
    dcc.Graph(id='state-job-graph')
])


# Set up the callback function
@app.callback(
    Output(component_id='state-job-graph', component_property='figure'),
    Input(component_id=geo_dropdown, component_property='value')
)
def update_graph(selected_state):
    filtered_state = jobs_info[jobs_info['Job Location'] == selected_state]
    line_fig = px.histogram(filtered_state,
                       x='Industry', y='Avg Salary(K)',
                       color='Type of ownership',
                       title=f'Average Salary in {selected_state}')
    return line_fig


# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)