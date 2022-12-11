# Import libraries

from dash import Dash, html, dcc, Input, Output , dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import dash_pivottable
from dash.exceptions import PreventUpdate
import pickle

# Load the dataset
jobs_info = pd.read_csv('Cleaned_Glassdoor_DS_Jobs.csv')
jobs_info.drop(jobs_info.loc[jobs_info['Industry']=='-1'].index, inplace=True)
jobs_info.drop(jobs_info.loc[jobs_info['Founded']==-1].index, inplace=True)

df_pivot_skills = jobs_info[['job_title_sim','Python', 'spark', 'aws', 'excel','sql','sas','keras','pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an','Avg Salary(K)']]

skills = ['Python', 'spark', 'aws', 'excel','sql','sas','keras','pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an']



#table = dbc.Table.from_dataframe(df_pivot_skills, striped=True, bordered=True, hover=True)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Create the Dash app
app = Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

ALLOWED_TYPES = (
   "number" 
)
# Set up the app layout
geo_dropdown = dcc.Dropdown(options=jobs_info['State'].unique(),
                            value='New York')
state_dropdown = dcc.Dropdown(options=jobs_info['State'].unique(),
                            placeholder= 'Select the State')                            
company_dropdown = dcc.Dropdown(options=jobs_info['Company Name'].unique(),
                            value='MITRE')    

skills_dropdown = dcc.Dropdown(options=skills, value= 'Python')

skills_dropdown2 = dcc.Dropdown(options=skills, placeholder='Choose your skills', multi= True)

df_rating_filter= jobs_info[jobs_info['Rating']!=-1]
Rating_slider = dcc.Slider(id = 'rating-slider', min= df_rating_filter['Rating'].min() , 
                                        max = df_rating_filter['Rating'].max(),
                                        value = 4.2  
                                        )
job_title_dropdown= dcc.Dropdown(options= jobs_info['Job Title'].unique(), placeholder= 'Select the job title')
salary_estimate_dropdown= dcc.Dropdown(options= jobs_info['Salary Estimate'].unique(), placeholder= 'Select your desired salary')
Location_dropdown= dcc.Dropdown(options= jobs_info['Location'].unique(), value= 'Jersey City, NJ')
Ownership_dropdown= dcc.Dropdown(options= jobs_info['Type of ownership'].unique(), placeholder= 'Select the type of company you want to work')
Industry_dropdown= dcc.Dropdown(options= jobs_info['Industry'].unique(), placeholder= 'Select the type of Industry you want to work at')
Sector_dropdown = dcc.Dropdown(options= jobs_info['Sector'].unique(), placeholder='Choose the sector')
Revenue_dropdown= dcc.Dropdown(options= jobs_info['Revenue'].unique(), placeholder='Choose the Revenuew of the anticipated company')
Hourly_radiobutton= dcc.RadioItems(options= ['Yes', 'No'] )
Lower_salary_text= dcc.Input(id="lsalary", type="number", placeholder="Lower Salary(K) ")   
Upper_salary_text= dcc.Input(id="usalary", type="number", placeholder="Upper Salary(K")
Age_of_company_text= dcc.Input(id="age_of_company",type= 'number', placeholder='Age of the company' )
Seniority_by_title_radiobutton= dcc.RadioItems(options=['Senior', 'Junior'])
Degree_radiobutton= dcc.RadioItems(options=['Masters', 'PHD', 'N/A'])
companies_dropdown= dcc.Dropdown(options=jobs_info['Company Name'].unique(),placeholder='Select the company')



app.layout = html.Div(children=[
    html.H1(children='Data Scientist Average Job Salary by States, Sectors, and Type of Ownership'),
    geo_dropdown,
    dcc.Graph(id='state-job-graph'),
    html.Br(),
    html.Div(children=[
        html.H1(children='Job Openings of Different Companies and their Location'),
        company_dropdown,
        dcc.Graph(id='company-job-graph')
    ]), 
    html.Div(children=[
        html.H1(children='Size of company and their ratings'),
        html.Label('Choose the Rating of the Company: '),
        Rating_slider , 
        html.Label('Choose the size of the Company: '),
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
             
    ]), 
    html.Br(),
    html.Div([ 
        html.H1(children='Make a prediction of your Average Salary'),
        html.Label('Select the job title'),
        job_title_dropdown,
        html.Label('Select the Company'),
        companies_dropdown,
        html.Label('Select your skills'),
        skills_dropdown2,
        html.Label('Select your estimated Salary'),
        salary_estimate_dropdown,
        html.Label('Select the State'),
        state_dropdown,
        html.Label('Choose the type of ownership'),
        Ownership_dropdown,
        html.Label('Choose the industry'),
        Industry_dropdown,
        html.Label('Choose the Sector'),
        Sector_dropdown,
        html.Label('Choose the Revenue'),
        Revenue_dropdown,
        html.Label('Paid Hourly?'),
        Hourly_radiobutton,
        html.Label('Input the Lower range of your desired salary(K)'),
        Lower_salary_text,
        html.Br(),
        html.Label('Input the Upper range of your desired salary(K)'),
        Upper_salary_text,
        html.Br(),
        html.Label('Input the age of the company you want to work at'),
        Age_of_company_text,
        html.Br(),
        html.Label('Choose your seniority at the job'),
        Seniority_by_title_radiobutton,
        html.Label('Choose your qualification'),
        Degree_radiobutton,
        html.Button('Predict', id='predict_sal', n_clicks=0),
        html.Div(id='predicted-value')
    ])
    
     ])
    
    
    



# Set up the callback function
@app.callback(
    Output(component_id='state-job-graph', component_property='figure'),
    
    Input(component_id=geo_dropdown, component_property='value'),
)
def update_graph(selected_state):
    filtered_state = jobs_info[jobs_info['State'] == selected_state]
    hist_fig = px.histogram(filtered_state,
                       x='Sector', y='Avg Salary(K)', barmode='stack',
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

@app.callback(
    Output(component_id='predicted-value', component_property='children'),
    [Input(component_id=job_title_dropdown, component_property= 'value'),
               Input(component_id=companies_dropdown, component_property= 'value'), 
               Input(component_id=skills_dropdown2, component_property= 'value'),
               Input(component_id=salary_estimate_dropdown, component_property= 'value'),
               Input(component_id=Ownership_dropdown, component_property= 'value'),
               Input(component_id=Industry_dropdown, component_property= 'value'),
               Input(component_id=Sector_dropdown, component_property= 'value'),
               Input(component_id=Revenue_dropdown, component_property= 'value'),
               Input(component_id=Hourly_radiobutton, component_property= 'value'),
               Input(component_id=Lower_salary_text, component_property= 'value'),
               Input(component_id=Upper_salary_text, component_property= 'value'),
               Input(component_id=Age_of_company_text, component_property= 'value'),
               Input(component_id=Seniority_by_title_radiobutton, component_property= 'value'),
               Input(component_id=Degree_radiobutton, component_property= 'value') ,
               Input(component_id='predict_sal', component_property='n_clicks')])
  
def predict (selected_job, selected_company, selected_skills, selected_estimated_salary, selected_ownership, selected_industry, 
selected_sector, selected_revenue, selected_hourly, selected_lower_salary, selected_upper_salary, selected_age, selected_seniority,
selected_degree, btn_click):
    print(selected_job, selected_company, selected_skills, selected_estimated_salary, selected_ownership, selected_industry, 
    selected_sector, selected_revenue, selected_hourly, selected_lower_salary, selected_upper_salary, selected_age, selected_seniority,
    selected_degree)
    file = open('DS_Jobs.pkl', 'rb')
    # dump information to that file
    data = pickle.load(file)
    skills_name = ['Python', 'spark', 'aws', 'excel','sql','sas','keras','pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an']
    skill_value=[ ]

    if 'predict_sal' == ctx.triggered_id:
        for i in range (16):
                if(skills_name[i] == selected_skills ):
                    skill_value.append(1)
                else:
                    skill_value.append(0)    

        print(skill_value)
        return "Clicked"
    else:
        raise PreventUpdate
    
  

# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)