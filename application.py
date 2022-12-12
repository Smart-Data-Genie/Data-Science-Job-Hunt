# Import libraries

from dash import Dash, html, dcc, Input, Output , dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
import pickle
import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import OrdinalEncoder
# Load the dataset
jobs_info = pd.read_csv('Cleaned_Glassdoor_DS_Jobs.csv')
HR_comma_sep= pd.read_csv('HR_comma_sep.csv')

jobs_info.drop(jobs_info.loc[jobs_info['Industry']=='-1'].index, inplace=True)
jobs_info.drop(jobs_info.loc[jobs_info['Founded']==-1].index, inplace=True)
jobs_info.drop(jobs_info.loc[jobs_info['Rating']==-1].index, inplace=True)

X = jobs_info.drop(columns=['Avg Salary(K)','Founded', 'Job Description', 'Headquarters', 'Location'
                     , 'Size', 'Employer provided', 'company_txt', 'Competitors', 'job_title_sim', 'Job Location', 'Revenue'
                    ,'Salary Estimate', 'Company Name', 'Sector', 'Age'])
print(X['Rating'].unique())
print(X.info())  

for column in X.columns:
            if X[column].dtype == type(object):
                le = sklearn.preprocessing.LabelEncoder()
                X[column]= le.fit_transform(X[column])
print(X)

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

Rating_slider = dcc.Slider(id = 'rating-slider', min= jobs_info['Rating'].min() , 
                                        max = jobs_info['Rating'].max(),
                                        value = 4.2  
                                        )
Rating_slider2 = dcc.Slider(id = 'rating-slider2', min= jobs_info['Rating'].min() , 
                                        max = jobs_info['Rating'].max()
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


Satisfaction_dropdown= dcc.Dropdown(options= HR_comma_sep['satisfaction_level'].unique(), placeholder='Choose the satisfaction_level')
last_evaluation_dropdown= dcc.Dropdown(options= HR_comma_sep['last_evaluation'].unique(), placeholder='Choose the last evluation of employee')
number_project_dropdown= dcc.Dropdown(options= HR_comma_sep['number_project'].unique(), placeholder='Choose the number of project done by the employee')
time_dropdown= dcc.Dropdown(options= HR_comma_sep['time_spend_company'].unique(), placeholder='Choose the time spend by the employee')
time_Work_accident_dropdown= dcc.Dropdown(options= HR_comma_sep['Work_accident'].unique(), placeholder='Choose the time work accident by the employee')
promotion_dropdown= dcc.Dropdown(options= HR_comma_sep['promotion_last_5years'].unique(), placeholder='How many promotions of the employee')
Department_dropdown= dcc.Dropdown(options= HR_comma_sep['sales'].unique(), placeholder='Choose the department of the employee')
Salary_dropdown= dcc.Dropdown(options= HR_comma_sep['salary'].unique(), placeholder='Choose the range of the employee''s salary')
Avg_Monthly_hours_dropdown= dcc.Dropdown(options= HR_comma_sep['average_montly_hours'].unique(), placeholder='Choose the average monthly hours of employee')

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
        html.Label('Select the rating of the company you want to work at:'),
        Rating_slider2,
        html.Label('Choose the type of ownership'),
        Ownership_dropdown,
        html.Label('Choose the industry'),
        Industry_dropdown,
        html.Label('Paid Hourly?'),
        Hourly_radiobutton,
        html.Label('Input the Lower range of your desired salary(K)'),
        Lower_salary_text,
        html.Br(),
        html.Label('Input the Upper range of your desired salary(K)'),
        Upper_salary_text,
        html.Br(),
        html.Label('Select your skills'),
        skills_dropdown2,
        html.Label('Choose your seniority at the job'),
        Seniority_by_title_radiobutton,
        html.Label('Choose your qualification'),
        Degree_radiobutton,
        html.Label('Select the State'),
        state_dropdown,
        html.Button('Predict', id='predict_sal', n_clicks=0),
        html.Br(),
        html.Div(id='predicted-value',   style={'color': 'blue', 'fontSize': 20})
    ],  style={'marginBottom': 50, 'marginTop': 25}),

        html.Br(),
    html.Div([ 
        html.H1(children='Make a prediction if the employee will leave the job'),
        html.Label('Select the satisfaction level of employee'),
        Satisfaction_dropdown,
        html.Label('Select the last year evaluation of the employee'),
        last_evaluation_dropdown,
        html.Label('Select the number of projects done by the employee'),
        number_project_dropdown,
        html.Label('Select the average monthly hours of the employe in the company'),
        Avg_Monthly_hours_dropdown  ,
        html.Label('Select the time spent  by the employe in the company'),
        time_dropdown,
        html.Label('Select the work accident done by the employe in the company'),
        time_Work_accident_dropdown, 
        html.Label('Was the employee promoted last year(1 for yes and 0 for No'),
        promotion_dropdown,
        html.Label('Select the department of the employee in the company'),
        Department_dropdown, 
        html.Label('Select the salary range of the employee in the company'),        
        Salary_dropdown,
        html.Button('Submit to Predict', id='predict_left', n_clicks=0),
        html.Br(),
        html.Div(id='predicted-value-employee',   style={'color': 'blue', 'fontSize': 20})
    ],  style={'marginBottom': 50, 'marginTop': 25})
    
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
    Input(component_id= 'rating-slider2', component_property= 'value') ,
    Input(component_id=Ownership_dropdown, component_property= 'value'),
    Input(component_id=Industry_dropdown, component_property= 'value'),
    Input(component_id=Hourly_radiobutton, component_property= 'value'),
    Input(component_id=Lower_salary_text, component_property= 'value'),
    Input(component_id=Upper_salary_text, component_property= 'value'),
    Input(component_id= skills_dropdown2, component_property= 'value'),
    Input(component_id=Seniority_by_title_radiobutton, component_property= 'value'),
    Input(component_id=Degree_radiobutton, component_property= 'value') ,
    Input (component_id= state_dropdown, component_property='value'),
    Input(component_id='predict_sal', component_property='n_clicks')])
  
def predict (selected_job, selected_rating, selected_ownership,  selected_industry, selected_hourly, selected_lower_salary, selected_upper_salary, 
 selected_skills,selected_seniority,
selected_degree, selected_state, btn_click):
    if (selected_hourly=='Yes'):
        selected_hourly =1
    else: 
        selected_hourly=0   
    print(selected_job, selected_rating, selected_ownership,  selected_industry, 
selected_hourly, selected_lower_salary, selected_upper_salary, selected_skills,selected_seniority,
selected_degree, selected_state, btn_click)
    file = open('DS_Jobs.pkl', 'rb')
    file2= open('RF_DS_Jobs.pkl', 'rb')
    file3 = open('DTC_DS_Jobs.pkl', 'rb')    
    # dump information to that file
    model1 = pickle.load(file2)
    model3= pickle.load(file3)
    skills_name = ['Python', 'spark', 'aws', 'excel','sql','sas','keras','pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an']
    Skill_value = np.zeros(16)

    if 'predict_sal' == ctx.triggered_id:
        

       #onehotencoder = sklearn.preprocessing.OneHotEncoder(sparse= False,  handle_unknown='ignore')
       #ohc= onehotencoder.fit(X)
       #print(ohc)
        for i in range (16):
            for j in range(len(selected_skills)):
                if(skills_name[i] == selected_skills[j] ):
                    Skill_value[i]=1

        print(Skill_value)
        print(type(selected_rating))
        print(type(selected_upper_salary))
        user1 = np.array( [selected_job, selected_rating, selected_ownership, selected_industry, 
          selected_hourly, selected_lower_salary, selected_upper_salary])
        print('User1', user1)

        print(type(user1[6]))
        user2= np.array([selected_seniority, selected_degree, selected_state])
        print('User2', user2)
        final_user_input= np.concatenate((user1,Skill_value,user2))
        print(final_user_input.shape)
        #final_user_input = np.reshape(final_user_input, (1, -1))
        #le.fit(range(max(final_user_input+1)))
       # df_final_input= pd.DataFrame(final_user_input, columns= ['Job Title', 'Rating', 'Type of Ownership', 'Industry', 'Hourly', 'Lower Salary', 'Upper Salary', 
        #'Python', 'spark', 'aws', 'excel','sql','sas','keras','pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an', 'seniority', 'Degree', 'State'])
        #print(df_final_input)
        final_input= le.fit_transform(final_user_input)

        #final_user_input= pd.get_dummies([final_user_input])
        #final_user_input = final_user_input.astype(np.float64)
        print(final_input)
        pred1 = model1.predict([final_input])
        pred3 = model3.predict([final_input])
        output1= abs(round(pred1[0],2))
        output2= abs(round(pred3[0],2))

        print('From Random Forest: ', output1)
        print('From Decision Tree Classifier: ', output2)
        return (f'The predicted salary is :${output1}K')
    else:
        raise PreventUpdate





@app.callback(
    Output(component_id='predicted-value-employee', component_property='children'),
    [Input(component_id=Satisfaction_dropdown, component_property= 'value'),
    Input(component_id= last_evaluation_dropdown, component_property= 'value') ,
    Input(component_id=number_project_dropdown, component_property= 'value'),
    Input(component_id=Avg_Monthly_hours_dropdown, component_property= 'value'),
    Input(component_id=time_dropdown, component_property= 'value'),
    Input(component_id=time_Work_accident_dropdown, component_property= 'value'),
    Input(component_id=promotion_dropdown, component_property= 'value'),
    Input(component_id=Department_dropdown, component_property= 'value'),
    Input(component_id= Salary_dropdown, component_property= 'value'),
    Input(component_id='predict_left', component_property='n_clicks')])
  
def predict_employee_retention (selected_satisfaction, selected_evaluation, selected_number_project_dropdown,  selected_monthly_hours,selected_time_dropdown, 
selected_accident, selected_promotion, selected_department, selected_salary, button_click):
    if (selected_department=='sales'):
        selected_department= 0
    elif(selected_department=='accounting') :
          selected_department=1
    elif(selected_department=='hr') :
          selected_department=2      
    elif(selected_department=='technical')  :
        selected_department= 3
    elif(selected_department=='support') :
          selected_department=4
    elif(selected_department=='management') :
          selected_department=5     
    elif(selected_department=='IT')  :
        selected_department= 6
    elif(selected_department=='product_mng')  :
        selected_department= 7
    elif(selected_department=='marketing') :
          selected_department=8
    else:
          selected_department=9    

    if (selected_salary=='low'):
        selected_salary= 0
    elif(selected_salary=='medium') :
          selected_salary=1
    elif(selected_salary=='high') :
          selected_salary=2      
    user_inputs= np.array([selected_satisfaction, selected_evaluation, selected_number_project_dropdown, selected_monthly_hours, selected_time_dropdown, 
selected_accident, selected_promotion, selected_department, selected_salary])
    print(selected_satisfaction, selected_evaluation, selected_number_project_dropdown, selected_monthly_hours, selected_time_dropdown, 
selected_accident, selected_promotion, selected_department, selected_salary,button_click )
    model_file = open('Job_Satf.pkl', 'rb')  
    # dump information to that file
    model_employee = pickle.load(model_file)

    if 'predict_left' == ctx.triggered_id:
        
        print(user_inputs)
        final_arr = user_inputs.astype(np.float64)
        pred1 = model_employee.predict([final_arr])
        output= abs(round(pred1[0],2))
        print (output)
        print('From Random Forest: ', output)
        if (output > 0 and output<0.5):
            return ('The employee is likely to stay in the company')
        elif(output >0.5 and output < 1):
            return ('The employee is likely to leave  the company')
    else:
        raise PreventUpdate 

# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)