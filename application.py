# Import libraries

from dash import Dash, html, dcc, Input, Output , dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import dash_pivottable

# Load the dataset
jobs_info = pd.read_csv('Cleaned_Glassdoor_DS_Jobs.csv')
jobs_info.drop(jobs_info.loc[jobs_info['Industry']=='-1'].index, inplace=True)
jobs_info.drop(jobs_info.loc[jobs_info['Founded']==-1].index, inplace=True)

df_pivot_skills = jobs_info[['job_title_sim','Python', 'spark', 'aws', 'excel','sql','sas','keras','pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an','Avg Salary(K)']]

skills = ['Python', 'spark', 'aws', 'excel','sql','sas','keras','pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an']
skills_dropdown = dcc.Dropdown(options=skills, value= 'Python')


#table = dbc.Table.from_dataframe(df_pivot_skills, striped=True, bordered=True, hover=True)

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
                            value='MITRE')    

df_rating_filter= jobs_info[jobs_info['Rating']!=-1]
Rating_slider = dcc.Slider(id = 'rating-slider', min= df_rating_filter['Rating'].min() , 
                                        max = df_rating_filter['Rating'].max(),
                                        value = 4.2  
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
            value = jobs_info['Size'].unique()[0:1],
            style={'float': 'middle','margin': 'auto'}
        ),
        dcc.Graph(id='size-rating-scatterplot')
    ]), 
    html.Div([
       html.H1(children='Skills Required by Companies for Each Job Title'), 
       html.Label('Choose the skills and see if they are required in different Job titles and see the difference in salaries'),
       skills_dropdown,
       html.Br(),
       html.Div(id="tableid")
             
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
    print(selected_company)
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
    print(selected_rating, selected_size)
    filtered_df = jobs_info.loc[(jobs_info['Rating'] == selected_rating) & (jobs_info['Size'].isin (selected_size ))]
    #filtered_df = jobs_info[jobs_info['Size'].isin (selected_size )]
    fig = px.scatter(filtered_df, x="Lower Salary", y="Upper Salary", 
        size="Avg Salary(K)", color="Job Title", hover_name="Company Name",
        log_x=True, 
        size_max=55, 
        ) 
    return fig

@app.callback(
    Output(component_id='tableid', component_property='children'),
    
    Input(component_id=skills_dropdown, component_property='value'),
)
def update_table(selected_skill):
    table = pd.pivot_table(df_pivot_skills, index = 'job_title_sim', columns = selected_skill, values = 'Avg Salary(K)', aggfunc = ['mean','count'])
    print (table)
    table= table.reset_index()
    print (table.columns)
    #return table.to_dict('rows')
    #datatable_col_list, datatable_data = datatable_settings_multiindex(table  
    table.columns = ['Job Titles', f'Average Salary(K) without { selected_skill} Skills', f'Average Salary(K) with {selected_skill} Skills',
     f'Job Description without {selected_skill} Skills', f'Job Description with {selected_skill} Skills']
    
    print(table)   
    print(table.columns) 

    #return dash_table.DataTable(data=datatable_data, columns=datatable_col_list)
    return dash_table.DataTable(data=table.to_dict('records'),columns=[{'name': i, 'id': i,} for i in table.columns])


# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)