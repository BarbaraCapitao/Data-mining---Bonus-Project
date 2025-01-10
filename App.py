import pandas as pd
import numpy as np
import os
from minisom import MiniSom
from sklearn.cluster import KMeans, AgglomerativeClustering
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import plotly.colors as colors

cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)  #color palette used throughout the notebook
fixed_color = cmap(0.5) #Selecting a color from the palette used throughout the notebook


# Define the cubehelix color palette
cubehelix_colors = sns.color_palette("cubehelix", 10).as_hex()  # Get 10 colors in hex format


# Load your dataset
df_clusters = pd.read_csv('customer_segmentation_clusters.csv')
print("Dataset loaded successfully.")

value_variables = ["product_count", "first_order", "last_order", "total_spent", "total_orders"]
preferences_variables = ["morning_orders_proportion", "lunch_orders_proportion", "afternoon_orders_proportion", "dinner_orders_proportion", "night_orders_proportion", "is_chain_proportion", "CUI_American_proportion", "CUI_Beverages_proportion", "CUI_Chicken Dishes_proportion", "CUI_Healthy_proportion", "CUI_Italian_proportion", "CUI_OTHER_proportion", "CUI_Street Food / Snacks_proportion", "CUI_Asian_Total_proportion"]
valid_customer_regions = [4660, 8670, 2360, 2440, 4140, 2490, 8370, 8550]

# KMeans Model
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df_clusters[value_variables])
print("KMeans model trained successfully.")

# SOM Parameters
M = 20
N = 20
neighborhood_function = 'gaussian'
topology = 'hexagonal'
n_feats = len(preferences_variables)
learning_rate = 0.7

# Initialize the MiniSom object
som = MiniSom(M, N, n_feats, learning_rate=learning_rate, topology=topology, neighborhood_function=neighborhood_function, activation_distance='euclidean', random_seed=42)
som_data = df_clusters[preferences_variables].values
som.random_weights_init(som_data)
som.train_batch(som_data, 500000)
print("SOM trained successfully.")

# SOM Weights for Hierarchical Clustering
weights_flat = som.get_weights().reshape((M * N), len(preferences_variables))
hclust = AgglomerativeClustering(linkage='ward', metric='euclidean', n_clusters=5).fit(weights_flat)
print("Hierarchical clustering trained successfully.")

# Cluster names
cluster_names = {
    "11": "Early Risers with a Snack Bias",
    "12": "Dinner Dwellers with Diverse Tastes",
    "13": "Italian Cuisine Connoisseurs",
    "14": "Nocturnal Nibblers",
    "15": "Chain Enthusiasts",
    "21": "Balanced Explorers",
    "22": "Lunch-Driven Loyalists",
    "23": "Afternoon Adventurers",
    "24": "Dessert Lovers with an Asian Twist",
    "25": "Beverage Buffs",
    "31": "Premium Users",
    "32": "Power Users",
    "33": "Italian Enthusiasts with Limited Chain Interest",
    "34": "Asian Cuisine Admirers",
    "35": "Loyal Chains with High Spending"
}

# Function to assign combined cluster labels
def assign_combined_cluster_labels(new_data):
    # Predict cluster from KMeans
    value_cluster = kmeans.predict(new_data[value_variables])[0]

    # Predict cluster from SOM + Hierarchical Clustering
    bmu_index = som.winner(new_data[preferences_variables].values[0])
    behavior_cluster = hclust.labels_[bmu_index[0] * M + bmu_index[1]]

    # Combine clusters (example logic: concatenation)
    final_cluster = f"{value_cluster + 1}{behavior_cluster + 1}"
    return final_cluster

# Initialize the Dash app
app = dash.Dash(__name__)

# Define dark and light mode layouts
dark_mode_layout = go.Layout(
    paper_bgcolor='rgb(17, 17, 17)',
    plot_bgcolor='rgb(17, 17, 17)',
    font=dict(color='white'),
    title=dict(font=dict(color='white')),
    xaxis=dict(title=dict(font=dict(color='white')), tickfont=dict(color='white')),
    yaxis=dict(title=dict(font=dict(color='white')), tickfont=dict(color='white'))
)

light_mode_layout = go.Layout(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(color='black'),
    title=dict(font=dict(color='black')),
    xaxis=dict(title=dict(font=dict(color='black')), tickfont=dict(color='black')),
    yaxis=dict(title=dict(font=dict(color='black')), tickfont=dict(color='black'))
)

# Define the layout of the app
app.layout = html.Div([
    html.H1(id="main-title", children="Customer Segmentation Dashboard"),
    
    # Theme toggle switch
    html.Div([
        html.Label(id="theme-label", children="Select Theme"),
        dcc.RadioItems(
            id='theme-switch',
            options=[
                {'label': 'Light Mode', 'value': 'light'},
                {'label': 'Dark Mode', 'value': 'dark'},
            ],
            value='light',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        )
    ], style={'padding': '10px', 'backgroundColor': 'lightgray'}),
    
    # Cluster Exploration Section
    html.Div([
        html.H2(id="cluster-title", children="Cluster Exploration"),
        
        # Dropdown for cluster selection
        html.Label(id="cluster-label", children="Select Cluster"),
        dcc.Dropdown(
            id='cluster-dropdown',
            options=[{'label': 'All', 'value': 'All'}] + [{'label': str(cluster), 'value': str(cluster)} for cluster in df_clusters['final_cluster'].unique()],
            value='All'
        ),
        
        # Slider for age range selection
        html.Label(id="age-range-label", children="Select Age Range"),
        dcc.RangeSlider(
            id='age-slider',
            min=df_clusters['customer_age'].min(),
            max=df_clusters['customer_age'].max(),
            value=[df_clusters['customer_age'].min(), df_clusters['customer_age'].max()],
            marks={i: str(i) for i in range(df_clusters['customer_age'].min(), df_clusters['customer_age'].max() + 1, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        
        # Bar chart for customer region distribution
        #dcc.Graph(id='region-bar-chart')
    ]),
    
    # Visualization Tools Section
    html.Div([
        html.H2(id="visualization-title", children="Visualization Tools"),
        
        # Bar chart for customer region distribution
        dcc.Graph(id='region-bar-chart'),

        # Heatmap for demographics
        dcc.Graph(id='demographics-heatmap'),
        
        # Tree map for preferred cuisines
        dcc.Graph(id='cuisines-tree-map'),

        # Line chart for order trends
        dcc.Graph(id='order-trend-line-chart'),

        # Box plot for customer spending
        dcc.Graph(id='spending-box-plot'),

        # Histogram for order frequency
        dcc.Graph(id='order-frequency-histogram'),

        # Pie chart for payment method distribution
        dcc.Graph(id='payment-method-pie-chart'),

        # Scatter plot for customer spending vs age
        dcc.Graph(id='spending-age-scatter-plot'),

        # Heatmap for active days
        dcc.Graph(id='active-days-heatmap'),

        # Bar chart for promotion impact
        dcc.Graph(id='promotion-impact-bar-chart')
    ]),
    
    # Cluster Prediction Section
    html.Div([
        html.H2(id="prediction-title", children="Cluster Prediction"),
        
        # Note about input values for proportions
        html.Div(children="Note: Input values for proportions need to be between 0 and 1.", style={'color': 'red', 'font-weight': 'bold', 'margin-bottom': '10px'}),
        
        # Input fields for prediction
        html.Div([
            html.Div([
                html.Label("Product Count"),
                dcc.Input(id='product-count-input', type='number', value=0)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("First Order"),
                dcc.Input(id='first-order-input', type='number', value=0)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Last Order"),
                dcc.Input(id='last-order-input', type='number', value=0)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Total Spent"),
                dcc.Input(id='total-spent-input', type='number', value=0)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Total Orders"),
                dcc.Input(id='total-orders-input', type='number', value=0)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Morning Orders Proportion"),
                dcc.Input(id='morning-orders-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Lunch Orders Proportion"),
                dcc.Input(id='lunch-orders-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Afternoon Orders Proportion"),
                dcc.Input(id='afternoon-orders-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Dinner Orders Proportion"),
                dcc.Input(id='dinner-orders-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Night Orders Proportion"),
                dcc.Input(id='night-orders-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("Is Chain Proportion"),
                dcc.Input(id='is-chain-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI American Proportion"),
                dcc.Input(id='CUI-American-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI Beverages Proportion"),
                dcc.Input(id='CUI-Beverages-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI Chicken Dishes Proportion"),
                dcc.Input(id='CUI-Chicken-Dishes-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI Healthy Proportion"),
                dcc.Input(id='CUI-Healthy-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI Italian Proportion"),
                dcc.Input(id='CUI-Italian-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI OTHER Proportion"),
                dcc.Input(id='CUI-OTHER-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI Street Food / Snacks Proportion"),
                dcc.Input(id='CUI-Street-Food-Snacks-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
            html.Div([
                html.Label("CUI Asian Total Proportion"),
                dcc.Input(id='CUI-Asian-Total-proportion-input', type='number', value=0, min=0, max=1)
            ], style={'display': 'inline-block', 'width': '48%', 'padding': '10px'}),
        ], style={'columnCount': 2}),
        
        # Button to trigger prediction
        html.Button('Predict Cluster', id='predict-button', n_clicks=0),
        
        # Result text output
        html.Div(id='prediction-result', children="Cluster Prediction Result: ")
])
    ])


# Define the callback to update the bar chart
@app.callback(
    Output('region-bar-chart', 'figure'),
    [Input('cluster-dropdown', 'value'),
     Input('age-slider', 'value'),
     Input('theme-switch', 'value')]
)
def update_bar_chart(selected_cluster, age_range, theme):
    # Filter the dataframe based on the selected cluster and age range
    if selected_cluster == 'All' or selected_cluster is None:
        filtered_df = df_clusters[(df_clusters['customer_age'] >= age_range[0]) & (df_clusters['customer_age'] <= age_range[1])]
    else:
        selected_cluster = int(selected_cluster)
        filtered_df = df_clusters[(df_clusters['final_cluster'] == selected_cluster) & (df_clusters['customer_age'] >= age_range[0]) & (df_clusters['customer_age'] <= age_range[1])]

    # Filter the DataFrame to only include valid customer regions
    filtered_df = filtered_df[filtered_df['customer_region'].isin(valid_customer_regions)]
    
    # Convert customer_region and final_cluster to string for categorical handling
    filtered_df['customer_region'] = filtered_df['customer_region'].astype(str)
    filtered_df['final_cluster'] = filtered_df['final_cluster'].astype(str)
    
    # Create the bar chart for customer_region
    region_counts = filtered_df['customer_region'].value_counts().reset_index()
    region_counts.columns = ['customer_region', 'count']
    fig = px.bar(
        region_counts, 
        x='customer_region', 
        y='count', 
        color='customer_region', 
        title='Distribution of Customer Region', 
        labels={'customer_region': 'Customer Region', 'count': 'Count'},
        color_discrete_sequence=cubehelix_colors  # Apply cubehelix palette
    )
    fig.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)
    return fig


# Define the callback to update the visualizations
@app.callback(
    [Output('main-title', 'style'),
     Output('theme-label', 'style'),
     Output('cluster-title', 'style'),
     Output('cluster-label', 'style'),
     Output('age-range-label', 'style'),
     Output('visualization-title', 'style'),
     Output('demographics-heatmap', 'figure'),
     Output('cuisines-tree-map', 'figure'),
     Output('order-trend-line-chart', 'figure'),
     Output('spending-box-plot', 'figure'),
     Output('order-frequency-histogram', 'figure'),
     Output('payment-method-pie-chart', 'figure'),
     Output('spending-age-scatter-plot', 'figure'),
     Output('active-days-heatmap', 'figure'),
     Output('promotion-impact-bar-chart', 'figure')],
    [Input('cluster-dropdown', 'value'),
     Input('age-slider', 'value'),
     Input('theme-switch', 'value')]
)
def update_visualizations(selected_cluster, age_range, theme):
    # Filter the dataframe based on the selected cluster and age range
    if selected_cluster == 'All' or selected_cluster is None:
        filtered_df = df_clusters[(df_clusters['customer_age'] >= age_range[0]) & (df_clusters['customer_age'] <= age_range[1])]
    else:
        selected_cluster = int(selected_cluster)
        filtered_df = df_clusters[(df_clusters['final_cluster'] == selected_cluster) & (df_clusters['customer_age'] >= age_range[0]) & (df_clusters['customer_age'] <= age_range[1])]
    
    # Filter the DataFrame to only include valid customer regions
    filtered_df = filtered_df[filtered_df['customer_region'].isin(valid_customer_regions)]
    
    # Convert customer_region and final_cluster to string for categorical handling
    filtered_df['customer_region'] = filtered_df['customer_region'].astype(str)
    filtered_df['final_cluster'] = filtered_df['final_cluster'].astype(str)
    
    # Define the title and label styles based on the selected theme
    title_style = {'color': 'white'} if theme == 'dark' else {'color': 'black'}
    label_style = {'color': 'white'} if theme == 'dark' else {'color': 'black'}
    
    # Create the heatmap for demographics
    demographics_heatmap = px.density_heatmap(
        filtered_df, 
        x='customer_region', 
        y='customer_age', 
        title='Heatmap of Customer Region vs Age',
        labels={'customer_region': 'Customer Region', 'customer_age': 'Customer Age'} #,
        #color_continuous_scale=px.colors.sequential.Cubehelix  # Apply cubehelix palettedem
    )
    demographics_heatmap.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)
    
    # Create the tree map for preferred cuisines
    cuisines_columns = [col for col in df_clusters.columns if col.startswith('CUI_') and not col.endswith('_proportion')]
    cuisines_data = filtered_df[cuisines_columns].sum().reset_index()
    cuisines_data.columns = ['Cuisine', 'Count']
    cuisines_data['Cuisine'] = cuisines_data['Cuisine'].str.replace('CUI_', '')

    cuisines_tree_map = px.treemap(
        cuisines_data, 
        path=['Cuisine'], 
        values='Count', 
        title='Preferred Cuisines'
        #,
        #color_discrete_sequence=px.colors.sequential.Cubehelix[:len(cuisines_data)]  # Sample discrete colors
    )
    cuisines_tree_map.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)

    # Create the line chart for order trends
    order_trend_line_chart = px.line(
        filtered_df, 
        x='first_order', 
        y='total_orders', 
        title='Order Trends Over Time',
        labels={'first_order': 'First Order Time', 'total_orders': 'Total Orders'},
        color_discrete_sequence=[cubehelix_colors[2]]  # Apply cubehelix palette
    )
    order_trend_line_chart.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)

    # Create the box plot for customer spending
    spending_box_plot = px.box(
        filtered_df, 
        x='final_cluster', 
        y='total_spent', 
        title='Customer Spending Across Clusters',
        labels={'final_cluster': 'Cluster', 'total_spent': 'Total Spent'},
        color_discrete_sequence=[cubehelix_colors[2]]  # Apply cubehelix palette
    )
    spending_box_plot.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)

    # Create the histogram for order frequency
    order_frequency_histogram = px.histogram(
        filtered_df, 
        x='total_orders', 
        title='Order Frequency Distribution',
        labels={'total_orders': 'Total Orders'},
        color_discrete_sequence=[cubehelix_colors[2]]  # Apply cubehelix palette
    )
    order_frequency_histogram.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)

    # Create the pie chart for payment method distribution
    payment_method_pie_chart = px.pie(
        filtered_df, 
        names='payment_method', 
        title='Payment Method Distribution',
        color_discrete_sequence=cubehelix_colors  # Apply cubehelix palette
    )
    payment_method_pie_chart.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)

    # Create the scatter plot for customer spending vs age
    spending_age_scatter_plot = px.scatter(
        filtered_df, 
        x='customer_age', 
        y='total_spent', 
        title='Customer Spending vs Age',
        labels={'customer_age': 'Customer Age', 'total_spent': 'Total Spent'},
        color_discrete_sequence=[cubehelix_colors[2]]  # Apply cubehelix palette
    )
    spending_age_scatter_plot.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)

    # Create the heatmap for active days
    active_days_heatmap = px.density_heatmap(
        filtered_df, 
        x='active_days', 
        y='final_cluster', 
        title='Heatmap of Active Days Across Clusters',
        labels={'active_days': 'Active Days', 'final_cluster': 'Cluster'}
        #,
        #color_continuous_scale=px.colors.sequential.Cubehelix  # Apply cubehelix palette
    )
    active_days_heatmap.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)

    # Create the bar chart for promotion impact
    promotion_impact_bar_chart = px.bar(
        filtered_df, 
        x='final_cluster', 
        y='had_promotion', 
        title='Promotion Impact Across Clusters',
        labels={'final_cluster': 'Cluster', 'had_promotion': 'Had Promotion'},
        color_discrete_sequence=[cubehelix_colors[2]]  # Apply cubehelix palette

    )
    promotion_impact_bar_chart.update_layout(dark_mode_layout if theme == 'dark' else light_mode_layout)
    
    return (title_style, title_style, title_style, label_style, label_style, title_style, 
            demographics_heatmap, cuisines_tree_map, order_trend_line_chart, 
            spending_box_plot, order_frequency_histogram, payment_method_pie_chart, 
            spending_age_scatter_plot, active_days_heatmap, promotion_impact_bar_chart)


# Define the callback to predict the cluster
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('product-count-input', 'value'),
     State('first-order-input', 'value'),
     State('last-order-input', 'value'),
     State('total-spent-input', 'value'),
     State('total-orders-input', 'value'),
     State('morning-orders-proportion-input', 'value'),
     State('lunch-orders-proportion-input', 'value'),
     State('afternoon-orders-proportion-input', 'value'),
     State('dinner-orders-proportion-input', 'value'),
     State('night-orders-proportion-input', 'value'),
     State('is-chain-proportion-input', 'value'),
     State('CUI-American-proportion-input', 'value'),
     State('CUI-Beverages-proportion-input', 'value'),
     State('CUI-Chicken-Dishes-proportion-input', 'value'),
     State('CUI-Healthy-proportion-input', 'value'),
     State('CUI-Italian-proportion-input', 'value'),
     State('CUI-OTHER-proportion-input', 'value'),
     State('CUI-Street-Food-Snacks-proportion-input', 'value'),
     State('CUI-Asian-Total-proportion-input', 'value')]
)
def predict_cluster(n_clicks, product_count, first_order, last_order, total_spent, total_orders, morning_orders_proportion, lunch_orders_proportion, afternoon_orders_proportion, dinner_orders_proportion, night_orders_proportion, is_chain_proportion, CUI_American_proportion, CUI_Beverages_proportion, CUI_Chicken_Dishes_proportion, CUI_Healthy_proportion, CUI_Italian_proportion, CUI_OTHER_proportion, CUI_Street_Food_Snacks_proportion, CUI_Asian_Total_proportion):
    if n_clicks > 0:
        new_data = pd.DataFrame({
            "product_count": [product_count],
            "first_order": [first_order],
            "last_order": [last_order],
            "total_spent": [total_spent],
            "total_orders": [total_orders],
            "morning_orders_proportion": [morning_orders_proportion],
            "lunch_orders_proportion": [lunch_orders_proportion],
            "afternoon_orders_proportion": [afternoon_orders_proportion],
            "dinner_orders_proportion": [dinner_orders_proportion],
            "night_orders_proportion": [night_orders_proportion],
            "is_chain_proportion": [is_chain_proportion],
            "CUI_American_proportion": [CUI_American_proportion],
            "CUI_Beverages_proportion": [CUI_Beverages_proportion],
            "CUI_Chicken Dishes_proportion": [CUI_Chicken_Dishes_proportion],
            "CUI_Healthy_proportion": [CUI_Healthy_proportion],
            "CUI_Italian_proportion": [CUI_Italian_proportion],
            "CUI_OTHER_proportion": [CUI_OTHER_proportion],
            "CUI_Street Food / Snacks_proportion": [CUI_Street_Food_Snacks_proportion],
            "CUI_Asian_Total_proportion": [CUI_Asian_Total_proportion]
        })
        final_cluster = assign_combined_cluster_labels(new_data)
        cluster_name = cluster_names.get(final_cluster, "Unknown Cluster")
        return f"Predicted Cluster: {final_cluster} - {cluster_name}"
    return "Cluster Prediction Result: "

# Run the Dash app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Get the PORT from environment variable
    app.run_server(debug=False, host="0.0.0.0", port=port)  # Use 0.0.0.0 to allow external connections
