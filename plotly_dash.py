from datetime import date, datetime, timedelta
from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def get_data(path='huge_excel.xlsx'):
    excel_df = pd.read_excel(path, index_col=0)
    excel_df = excel_df.pivot_table(index=['pricing_date', 'order_date'], columns='market_maker_mnemonic', values='daily_quantity').fillna(0)
    excel_df.reset_index(inplace=True)
    excel_df.set_index('pricing_date', inplace=True)

    excel_df.index = pd.to_datetime(excel_df.index)
    excel_df.order_date = pd.to_datetime(excel_df.order_date)
    return excel_df


data = get_data()
fmt = '%Y-%m-%d'

start_date = data.index.max().year
end_date = f'{start_date}-12-31'
start_date = f'{int(start_date) - 1}-01-01'

START_DATE, END_DATE = start_date, end_date

MARKS = None

print('end_date:', end_date)
print('start_date:', start_date)


def get_dates(df, ds, de):
    temp = df[(df.order_date >= ds) & (df.order_date <= de)]   
    return 0, len(temp.order_date.unique()), get_marks(temp)


def get_marks(temp):
    global MARKS
    marks = {}
    
    for idx, mark in enumerate(temp.order_date.unique()):
        marks[idx] = {'label': str(mark)[:10]}
        
    MARKS = marks
    
    return marks


def create_color_dict(columns):
    colors = px.colors.qualitative.Alphabet
    color_dict = {column: colors[i % len(colors)] for i, column in enumerate(columns)}
    return color_dict


color_dict = create_color_dict(['ALVARI', 'ARAMCOSG', 'ARAMCOTF', 'ARCENERGY', 'BBEN', 'BPSG',
                                'BRIGHTOILSG', 'BUYER1', 'BUYER2', 'CAOSG', 'CARGILLSG', 'CCMA',
                                'CHEVRONSG', 'COASTAL', 'ENEOSSG', 'ENOC', 'FREEPTSG', 'GLENCORESG',
                                'GPGLOBALSG', 'GULFSG', 'GUNVORSG', 'HL', 'IDEMITSU', 'ITGRES',
                                'KAIROS', 'KOCHRI', 'LUKOIL', 'MACQUARIESG', 'MAERSKSG',
                                'MERCURIARESOURCES', 'MERCURIASG', 'METS', 'MIPCO', 'P66SG', 'PETCO',
                                'PETROCHINA', 'PETROSUMMIT', 'PTT', 'REPSOLSG', 'REXCOMM', 'RGES',
                                'SELLER1', 'SIETCO', 'SINOHKPET', 'SINOPECFO', 'SINOPECHKSG', 'SKEISG',
                                'SOCAR', 'SUMMITENERGY', 'TOTALSG', 'TRAFI', 'UNIPECSG', 'VITOLSG',
                                'WANXIANG', 'ZENROCK'])


app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.Label('Select order_date start/end date'),
        html.Br(),
        dcc.DatePickerRange(
            id='order_end',
            min_date_allowed=datetime.strptime(start_date, fmt),
            max_date_allowed=datetime.strptime(end_date, fmt),
            end_date=datetime.strptime(end_date, fmt),
            start_date=datetime.strptime(end_date, fmt) - timedelta(days=10),
            initial_visible_month=datetime.strptime(str(data.index.max())[:10], fmt),
        ),
    ], style={'padding': 10, 'flex': 1}),
    
    html.Div(id='choose_order_date'),
    
    html.Div([
        html.Label('Select pricing_date span'),
        dcc.Slider(0, 90, 30, value=0, id='pricing_date'),
    ], style={'padding': 10, 'flex': 1}),
    
    html.Div([
        html.Label('Select order_date history lookback'),
        html.Br(),
        dcc.Input(id='order_lookback', type='number', value=0, placeholder='Select order_date history lookback', style={'padding': '10px'}),
    ], style={'padding': 10, 'flex': 1}),
    
    dcc.Graph(id='output-graph')
])

@app.callback(
    Output('choose_order_date', 'children'),
    [Input('order_end', 'start_date'),  # order_date start date
     Input('order_end', 'end_date')]  # order_date end date
)
def update_slider(start, end):
    global START_DATE, END_DATE
    
    START_DATE = start
    END_DATE = end
    
    widget = [html.Label('Choose order_date available option')]
    temp = data[(data.order_date > start) & (data.order_date < end)]
    widget.append(dcc.Slider(0, len(temp.order_date.unique()), 1, value=0, marks=get_marks(temp), included=False, id='show_order_date'))
    return widget

@app.callback(
    Output('output-graph', 'figure'),
    [Input('order_end', 'start_date'),  # order_date start date
     Input('order_end', 'end_date'),  # order_date end date
     Input('choose_order_date', 'children'),  # slider to select order_date to display
     Input('order_lookback', 'value'),  # slider to select the lookback (make it lighter)
     Input('pricing_date', 'value')]
)
def update_graph(start, end, current_od, lookback, pricing_days):
    global MARKS, START_DATE, END_DATE

    if start is not None and end is not None and current_od is not None and lookback is not None:
        current_od = current_od[1]['props']['value']

        price_date = datetime.strptime(end, fmt) + timedelta(days=pricing_days)
        price_date = str(price_date)[:10]
        print('price_date:', price_date)

        print(data.head())

        temp = data[(data.order_date >= START_DATE) & (data.order_date <= END_DATE)]

        # update MARKS
        if START_DATE != start or END_DATE != end:
            MARKS = None
            marks = get_marks(temp)

        print('TEMP:', temp)

        print('MARKS:', MARKS)

        temp = temp[temp.order_date == MARKS[current_od]['label']]
        print('TEMP HEAD:', temp.head())

        # Create subplot with two stacked bar charts
        fig = make_subplots(rows=1, cols=1)

        fig1 = px.bar(temp[temp.index <= price_date].drop(columns='order_date'), color_discrete_map=color_dict, text_auto=True)

        for trace in fig1.data:
            fig.add_trace(trace)

        if lookback > 0:
            new_date = datetime.strptime(str(temp.order_date.min())[:10], fmt) - timedelta(days=lookback)
            new_date = str(new_date)[:10]

            temp = data[(data.order_date >= new_date) & (data.order_date <= str(END_DATE)[:10])]

            fig2 = px.bar(temp[temp.index <= price_date].drop(columns='order_date'), color_discrete_map=color_dict, text_auto=True, opacity=0.4)

            for trace in fig2.data:
                fig.add_trace(trace)

        fig.update_layout(barmode='stack')
        return fig

    return px.bar(data.drop(columns='order_date'))


if __name__ == '__main__':
    app.run(debug=True)




# from datetime import date, datetime, timedelta
# from dash import Dash, dcc, html, Input, Output, callback
# import pandas as pd
# import plotly.express as px
# from plotly.subplots import make_subplots

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# def get_data(path='huge_excel.xlsx'):
#     excel_df = pd.read_excel(path, index_col=0)
#     excel_df = excel_df.pivot_table(index=['pricing_date', 'order_date'], columns='market_maker_mnemonic', values='daily_quantity').fillna(0)
#     excel_df.reset_index(inplace=True)
#     excel_df.set_index('pricing_date', inplace=True)

#     excel_df.index = pd.to_datetime(excel_df.index)
#     excel_df.order_date = pd.to_datetime(excel_df.order_date)
#     return excel_df


# data = get_data()
# fmt = '%Y-%m-%d'

# # start_date = datetime.strptime(str(data.index.max())[:10], fmt)
# # end_date = start_date - timedelta(days=365)
# # end_date = str(pd.to_datetime(end_date))[:10]

# start_date = data.index.max().year
# end_date = f'{start_date}-12-31'
# start_date = f'{int(start_date) - 1}-01-01'

# START_DATE, END_DATE = start_date, end_date

# MARKS = None

# print('end_date:', end_date)
# print('start_date:', start_date)


# def get_dates(df, ds, de):
#     # ds = date start, de = date end
    
#     temp = df[(df.order_date >= ds) & (df.order_date <= de)]   
    
#     return 0, len(temp.order_date.unique()), get_marks(temp)


# def get_marks(temp):
#     global MARKS
    
#     marks = {}
    
#     for idx, mark in enumerate(temp.order_date.unique()):
#         marks[idx] = {'label': str(mark)[:10]}
        
    
#     MARKS = marks
    
#     return marks


# def create_color_dict(columns):
#     # Predefined set of colors
#     colors = px.colors.qualitative.Alphabet

#     # Create a unique color dictionary for the given columns
#     color_dict = {column: colors[i % len(colors)] for i, column in enumerate(columns)}

#     return color_dict


# color_dict = create_color_dict(['ALVARI', 'ARAMCOSG', 'ARAMCOTF', 'ARCENERGY', 'BBEN', 'BPSG',
#                     'BRIGHTOILSG', 'BUYER1', 'BUYER2', 'CAOSG', 'CARGILLSG', 'CCMA',
#                     'CHEVRONSG', 'COASTAL', 'ENEOSSG', 'ENOC', 'FREEPTSG', 'GLENCORESG',
#                     'GPGLOBALSG', 'GULFSG', 'GUNVORSG', 'HL', 'IDEMITSU', 'ITGRES',
#                     'KAIROS', 'KOCHRI', 'LUKOIL', 'MACQUARIESG', 'MAERSKSG',
#                     'MERCURIARESOURCES', 'MERCURIASG', 'METS', 'MIPCO', 'P66SG', 'PETCO',
#                     'PETROCHINA', 'PETROSUMMIT', 'PTT', 'REPSOLSG', 'REXCOMM', 'RGES',
#                     'SELLER1', 'SIETCO', 'SINOHKPET', 'SINOPECFO', 'SINOPECHKSG', 'SKEISG',
#                     'SOCAR', 'SUMMITENERGY', 'TOTALSG', 'TRAFI', 'UNIPECSG', 'VITOLSG',
#                     'WANXIANG', 'ZENROCK'])

    
# app = Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div([
#     html.Div([
#         html.Label('Select order_date start/end date'),
#         html.Br(),
#         dcc.DatePickerRange(
#             id='order_end',
#             min_date_allowed=datetime.strptime(start_date, fmt),
#             max_date_allowed=datetime.strptime(end_date, fmt),
#             end_date=datetime.strptime(end_date, fmt),
#             start_date=datetime.strptime(end_date, fmt) - timedelta(days=10),
#             initial_visible_month=datetime.strptime(str(data.index.max())[:10], fmt),
#     ),
#     ], style={'padding': 10, 'flex': 1}),
    
#     html.Div(id='choose_order_date'),
    
#     # html.Div([
#     #     # html.Label('Choose order_date available option'),
#     #     # dcc.Slider(0, len(data[data.index < end_date].order_date.unique()), 1, value=0, marks=get_marks(data), included=False, id='show_order_date')
    
#     # ]),
    
#     html.Div([
#         html.Label('Select pricing_date span'),
#         dcc.Slider(0, 90, 30, value=0, id='pricing_date'),
#         # html.Br(),
#         # dcc.Input(id='pricing_date', type='number', value=30, placeholder='Select pricing_date span', style={'padding':'10px'})
        
#     ], style={'padding': 10, 'flex': 1}),
    
#     html.Div([
#         html.Label('Select order_date history lookback'),
#         html.Br(),
#         dcc.Input(id='order_lookback', type='number', value=0, placeholder='Select order_date history lookback', style={'padding':'10px'})
    
#     ], style={'padding': 10, 'flex': 1}),
    
#     dcc.Graph(id='output-graph')

#     # html.Div(id='output-container-date-picker-single'),
    
# ])

# @callback(
#     Output('choose_order_date', 'children'),
#     Input('order_end', 'start_date'),    # order_date start date
#     Input('order_end', 'end_date'),      # order_date end date
# )
# def update_slider(start, end):
#     widget = [html.Label('Choose order_date available option')]
#     temp = data[(data.order_date > start) & (data.order_date < end)]
    
#     widget.append(dcc.Slider(0, len(temp.order_date.unique()), 1, value=0, marks=get_marks(temp), included=False, id='show_order_date'))
    
#     return widget

# @callback(
#     Output('output-graph', 'figure'),
#     Input('order_end', 'start_date'),    # order_date start date
#     Input('order_end', 'end_date'),      # order_date end date
#     Input('choose_order_date', 'show_order_date'),   # slider to select order_date to display
#     Input('order_lookback', 'value'),    # slider to select the lookback (make it lighter)
#     Input('pricing_date', 'value')
# )
# def update_graph(start, end, current_od, lookback, pricing_days):
#     global MARKS, START_DATE, END_DATE
    
#     if start is not None and end is not None and current_od is not None and lookback is not None: 
              
#         print('end:', end)
#         print('pricing_days:', pricing_days)
        
#         price_date = datetime.strptime(end, fmt) + timedelta(days=pricing_days)
#         price_date = str(price_date)[:10]
#         print('price_date:', price_date)
        
#         print(data.head())
        
#         temp = data[(data.order_date >= start_date) & (data.order_date <= end_date)]
        
#         # update MARKS
#         if START_DATE != start or END_DATE != end:
#             MARKS, START_DATE, END_DATE = None, None, None
#             marks = get_marks(temp)
        
#         print('TEMP:', temp)
        
#         print('MARKS:', MARKS)
        
#         temp = temp[temp.order_date == MARKS[current_od]['label']]
#         print('TEMP HEAD:', temp.head())
        
#         # fig = px.bar(temp.drop(columns='order_date'))
        
#         print(temp[temp.index <= price_date].drop(columns='order_date'))
        
#         # Create subplot with two stacked bar charts
#         fig = make_subplots(rows=1, cols=1)

#         fig1 = px.bar(temp[temp.index <= price_date].drop(columns='order_date'), color_discrete_map=color_dict, text_auto=True)

#         for trace in fig1.data:
#             fig.add_trace(trace)
        
#         if lookback > 0:
#             new_date = datetime.strptime(str(temp.order_date.min())[:10], fmt) - timedelta(days=lookback)
#             new_date = str(new_date)[:10]
            
#             temp = data[(data.order_date >= new_date) & (data.order_date <= end_date)]
            
#             fig2 = px.bar(temp[temp.index <= price_date].drop(columns='order_date'), color_discrete_map=color_dict, text_auto=True, opacity=0.4)
            
#             for trace in fig2.data:
#                 fig.add_trace(trace)
            
#         fig.update_layout(barmode='stack')
#         return fig
    
#     return px.bar(data.drop(columns='order_date'))



# if __name__ == '__main__':
#     app.run(debug=True)

# @callback(
#     Output('output-graph', 'figure'),
#     Input('order_end', 'start_date'),
#     Input('order_end', 'end_date'),
#     Input('order_lookback', 'value'),
#     Input('pricing_date', 'value')
# )
# def update_plot(order_start, order_end, order_lookback, pricing_date):
    
#     if order_start is not None and order_end is not None and order_lookback is not None and pricing_date is not None:        
#         order_end = str(order_end)[:10]

#         new_date = datetime.strptime(order_end, fmt) - timedelta(days=order_lookback)
        
#         print('new date:', new_date)
        
#         data.order_date = pd.to_datetime(data.order_date)
        
#         print(str(pd.to_datetime(new_date))[:10])
        
#         filt = data.order_date > str(pd.to_datetime(new_date))[:10]
        
#         print('\n\n\nnew date:', new_date, end='\n\n\n')
#         print(data[filt].drop(columns='order_date'))
        
#         return px.bar(data[filt].drop(columns='order_date'))
    
    
#     fig = px.bar(data.drop(columns='order_date'))
#     return fig


# @callback(
#     Output('output-container-date-picker-single', 'children'),
#     Input('my-date-picker-single', 'date'))
# def update_output(date_value):
#     string_prefix = 'You have selected: '
#     if date_value is not None:
#         date_object = date.fromisoformat(date_value)
#         date_string = date_object.strftime('%B %d, %Y')
#         return string_prefix + date_string



# app.layout = html.Div([
#     dcc.DatePickerRange(
#         id='my-date-picker-range',
#         min_date_allowed=date(1995, 8, 5),
#         max_date_allowed=date(2017, 9, 19),
#         initial_visible_month=date(2017, 8, 5),
#         end_date=date(2017, 8, 25)
#     ),
#     html.Div(id='output-container-date-picker-range')
# ])


# @callback(
#     Output('output-container-date-picker-range', 'children'),
#     Input('my-date-picker-range', 'start_date'),
#     Input('my-date-picker-range', 'end_date'))
# def update_output(start_date, end_date):
#     string_prefix = 'You have selected: '
#     if start_date is not None:
#         start_date_object = date.fromisoformat(start_date)
#         start_date_string = start_date_object.strftime('%B %d, %Y')
#         string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
#     if end_date is not None:
#         end_date_object = date.fromisoformat(end_date)
#         end_date_string = end_date_object.strftime('%B %d, %Y')
#         string_prefix = string_prefix + 'End Date: ' + end_date_string
#     if len(string_prefix) == len('You have selected: '):
#         return 'Select a date to see it displayed here'
#     else:
#         return string_prefix


# if __name__ == '__main__':
#     app.run(debug=True)