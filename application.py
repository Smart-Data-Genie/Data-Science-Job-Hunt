# Import libraries

from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
jobs_info = pd.read_csv('Cleaned_Glassdoor_DS_Jobs.csv')
jobs_info.drop(jobs_info.loc[jobs_info['Industry']=='-1'].index, inplace=True)
jobs_info.drop(jobs_info.loc[jobs_info['Founded']==-1].index, inplace=True)

# Create the Dash app
app = Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
# Set up the app layout
geo_dropdown = dcc.Dropdown(options=jobs_info['State'].unique(),
                            value='New York')
company_dropdown = dcc.Dropdown(options=jobs_info['Company Name'].unique(),
                            value='Affinity Solutions')    

df_rating_filter= jobs_info[jobs_info['Rating']!=-1]
Rating_slider = dcc.Slider(id = 'rating-slider', min= df_rating_filter['Rating'].min() , 
                                        max = df_rating_filter['Rating'].max(),
                                        value = df_rating_filter['Rating'].min(),
                                        marks = { str(Rating) : str(Rating) for Rating in df_rating_filter['Rating'].unique() },
                                        step = None
                                        )

app.layout = html.Div(children=[
    html.H1(children='Data Scientist Average Job Salary by States, Sectors, and Type of Ownership'),
    geo_dropdown,
    dcc.Graph(id='state-job-graph'),
    
    html.Div(children=[
        html.H1(children='Job Openings of Different Companies and their Location'),
        company_dropdown,
        dcc.Graph(id='company-job-graph')
    ]), 
    html.Div(children=[
        html.H1(children='Size of company and their ratings'),
        html.Label('Choose the Rating of the Company: '),
        Rating_slider , 
        html.Label('Choose the size of the Company: ', style={'float': 'left','margin': 'auto'}),
        dcc.Checklist( id= 'Size_check_list',
            options= jobs_info['Size'].unique(),
            value =['1 - 50 '], style={'float': 'middle','margin': 'auto'}
        ),
        dcc.Graph(id='size-rating-scatterplot')
    ]) ])
    
    
    



# Set up the callback function
@app.callback(
    Output(component_id='state-job-graph', component_property='figure'),
    
    Input(component_id=geo_dropdown, component_property='value'),
)
def update_graph(selected_state):
    filtered_state = jobs_info[jobs_info['State'] == selected_state]
    hist_fig = px.bar(filtered_state,
                       x='Sector', y='Avg Salary(K)', barmode='group',
                       color='Type of ownership', 
                       title=f'Maximum of Avg Salary in {selected_state}')#.update_layout(yaxis_title="Maximum of Average Salary")
    hist_fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)
    return hist_fig



@app.callback(
    Output(component_id='company-job-graph', component_property='figure'),
    Input(component_id= company_dropdown, component_property='value'))

def update_graph2(selected_company):
    filtered_company = jobs_info[jobs_info['Company Name'] == selected_company]
    hist_fig2 = px.histogram(filtered_company,
                    x= 'Location',
                    barmode='group',
                    color='Job Title',
                    title=f'Different Job openings in {selected_company}').update_layout(yaxis_title="Number of Jobs")
    return hist_fig2    


# Set up the callback function
@app.callback(
    Output(component_id='size-rating-scatterplot', component_property='figure'),
    [Input(component_id='rating-slider', component_property= 'value'),
               Input(component_id='Size_check_list', component_property= 'value')])
  
def update_Scatter_plot (selected_rating, selected_size):
    filtered_df = jobs_info.loc[(jobs_info['Rating']==selected_rating) & (jobs_info['Size'] == selected_size)]
    fig = px.scatter(filtered_df, x="Company Name", y="Uppr Salary", 
        size="pop", color="Lower Salary", hover_name="Company Name",
        log_x=True, 
        size_max=55, 
        )

      
    return fig


# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)